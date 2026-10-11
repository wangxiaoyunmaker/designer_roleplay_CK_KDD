"""V13.8 V7-only localized Step Realizer trainer.

No Q2 paragraph, auxiliary response, Q7 material, or Composer output is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from route_a.data.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from route_a.inference.stepwise_realizer import first_sentence, render_step_prompt, step_format_failure, step_template_risk
from route_a.state_ontology import load_kc_state_contract
from route_a.training.config import load_training_config
from route_a.training.runtime import directory_sha256, load_tokenizer_and_causal_lm, require_training_runtime

STATE_ORDER, STATE_CONTRACT = load_kc_state_contract()
NEGATIVE = {"CtoK": "KtoC", "KtoC": "CtoK", "CtoC": "KtoK", "KtoK": "CtoC", "C": "K", "K": "C"}
NEIGHBOR_MASK = "—"
STEP_GENERATION_MAX_NEW_TOKENS = 45


class StepTrainingError(RuntimeError):
    pass


def load_step_config(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    required = {"schema_version", "experiment_id", "base_training_config", "data", "lora", "training", "selection", "forbid_q7"}
    if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != "1.0.0" or raw["forbid_q7"] is not True:
        raise StepTrainingError("invalid V13.8 step config")
    training = raw["training"]
    if training["lambda_state"] < 0 or training["margin"] < 0 or training.get("lambda_v30", 0) < 0 or training.get("v30_margin", 0) < 0:
        raise StepTrainingError("invalid state contrast settings")
    return raw


def validate_rows(rows: list[dict[str, Any]], *, formal: bool) -> Counter[str]:
    if not rows:
        raise StepTrainingError("empty V7 step split")
    coverage = Counter(row.get("current_state") for row in rows)
    for row in rows:
        if row.get("schema_version") != "v7_step_realizer_v1" or row.get("q7_gold_used") is not False or row.get("pi_used") is not False:
            raise StepTrainingError("V7-only/Q7-free record contract failed")
        if not (row.get("human_verified") and row.get("state_match_verified") and row.get("order_verified")):
            raise StepTrainingError("unverified V7 step cannot enter SFT")
        if row.get("current_state") not in STATE_ORDER:
            raise StepTrainingError("unsupported state")
        if row.get("segment_role") not in {"initial", "process", "boundary"} or not isinstance(row.get("segment_task"), str):
            raise StepTrainingError("V13.9 step record lacks a valid segment task")
        if len(str(row.get("compact_value_profile", "")).splitlines()) != 30:
            raise StepTrainingError("V13.9 step record lacks a complete compact V30 profile")
        if not (8 <= len(row.get("value_cues", [])) <= 10) or len(row.get("active_value_item_ids", [])) != len(row.get("value_cues", [])):
            raise StepTrainingError("V13.9 step record has invalid active V30 cues")
        if not isinstance(row.get("target_text"), str) or not (15 <= len("".join(row["target_text"].split())) <= 45):
            raise StepTrainingError("invalid step target length")
    if formal and any(coverage[state] == 0 for state in STATE_ORDER):
        raise StepTrainingError("INSUFFICIENT_COVERAGE: formal training requires all six states")
    return coverage


def validate_instrument_aux_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    coverage = Counter(row.get("current_state") for row in rows)
    for row in rows:
        if row.get("schema_version") != "instrument_state_aux_v1" or row.get("task_role") != "instrument_state_aux":
            raise StepTrainingError("invalid instrument auxiliary record")
        if row.get("q7_gold_used") is not False or row.get("pi_used") is not False or row.get("legacy_option_annotation") is not True:
            raise StepTrainingError("instrument auxiliary provenance contract failed")
        if row.get("current_state") not in STATE_ORDER or not isinstance(row.get("target_text"), str) or not (12 <= len("".join(row["target_text"].split())) <= 90):
            raise StepTrainingError("invalid instrument auxiliary state/text")
    return coverage


def counterfactual_row(row: dict[str, Any], wrong_state: str | None = None) -> dict[str, Any]:
    result = dict(row); result["current_state"] = wrong_state or NEGATIVE[row["current_state"]]
    # Only the current-state semantics may change. All other conditioning stays fixed.
    result["current_state_definition"] = STATE_CONTRACT[result["current_state"]]["definition"]
    return result


def negative_states(state: str) -> list[str]:
    """The five wrong states for a full 6-way state contrast."""
    return [s for s in STATE_ORDER if s != state]


def masked_state_row(row: dict[str, Any], state: str) -> dict[str, Any]:
    """State variant with the neighbouring states masked to a neutral token.

    Holding the true ``previous_state``/``next_state`` fixed while only flipping
    ``current_state`` lets a model satisfy the state-permutation contrast by
    reading the neighbours (in a real KC sequence the neighbours largely
    determine the current state) instead of the current-state field itself.
    Masking both sides makes the current-state field the *only* signal that
    distinguishes the correct prompt from each wrong one, so the gate certifies
    that the model actually reads the state it is asked to realise.
    """
    result = dict(row)
    result["current_state"] = state
    result["current_state_definition"] = STATE_CONTRACT[state]["definition"]
    result["previous_state"] = NEIGHBOR_MASK
    result["next_state"] = NEIGHBOR_MASK
    return result


def v30_counterfactual_row(row: dict[str, Any], candidates: Sequence[dict[str, Any]], rng: random.Random) -> dict[str, Any] | None:
    """Swap the complete V30 representation in the same local KC/task context."""
    own_cues = tuple(row.get("value_cues", ()))
    eligible = [
        candidate for candidate in candidates
        if candidate.get("participant_id") != row.get("participant_id")
        and candidate.get("current_state") == row.get("current_state")
        and candidate.get("position") == row.get("position")
        and candidate.get("segment_role") == row.get("segment_role")
        and tuple(candidate.get("value_cues", ())) != own_cues
    ]
    if not eligible:
        return None
    source = rng.choice(eligible)
    result = dict(row)
    result["value_cues"] = list(source["value_cues"])
    result["compact_value_profile"] = source["compact_value_profile"]
    result["active_value_item_ids"] = list(source["active_value_item_ids"])
    result["v30_counterfactual_source_participant_id"] = source["participant_id"]
    return result


def state_balanced_indices(rows: list[dict[str, Any]], *, count: int, rng: random.Random) -> list[int]:
    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): by_state[row["current_state"]].append(index)
    active = sorted(by_state)
    if not active: raise StepTrainingError("no active states")
    return [rng.choice(by_state[active[index % len(active)]]) for index in range(count)]


def stratified_evaluation_rows(rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    """Round-robin held-out examples so every state is auditable at each eval.

    The previous prefix slice could contain one CtoC example, making a
    per-state top-share of 1.0 statistically meaningless.  This deterministic
    sampler avoids changing the participant-held-out split while making the
    free-generation gate report all six state behaviours.
    """
    if count <= 0:
        return []
    buckets = {state: [row for row in rows if row.get("current_state") == state] for state in STATE_ORDER}
    selected: list[dict[str, Any]] = []
    cursor = {state: 0 for state in STATE_ORDER}
    while len(selected) < count:
        added = False
        for state in STATE_ORDER:
            index = cursor[state]
            if index < len(buckets[state]) and len(selected) < count:
                selected.append(buckets[state][index])
                cursor[state] += 1
                added = True
        if not added:
            break
    return selected


def _chat_ids(tokenizer: Any, prompt: str, target: str, device: Any) -> tuple[Any, Any]:
    torch = __import__("torch")
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    suffix = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + suffix], dtype=torch.long, device=device)
    labels = ids.clone(); labels[:, :len(prefix)] = -100
    return ids, labels


def _nll(model: Any, tokenizer: Any, row: dict[str, Any], device: Any) -> Any:
    ids, labels = _chat_ids(tokenizer, render_step_prompt(row), row["target_text"], device)
    return model(input_ids=ids, labels=labels, use_cache=False).loss


def _evaluate(
    torch: Any, model: Any, tokenizer: Any, rows: list[dict[str, Any],], device: Any,
    max_examples: int, *, require_v30: bool, max_template_risk: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = stratified_evaluation_rows(rows, count=max_examples)
    model.eval(); correct, wrong, value_pairs, outputs = [], [], [], []
    margins6, argmin_hits, pid_margins = [], [], []
    argmin_confusion = {t: Counter() for t in STATE_ORDER}
    rng = random.Random(0)
    with torch.no_grad():
        for row in selected:
            true = row["current_state"]
            correct_nll = float(_nll(model, tokenizer, row, device).item())
            correct.append(correct_nll)
            # Legacy single-negative contrast (real neighbours) — diagnostic only.
            wrong.append(float(_nll(model, tokenizer, counterfactual_row(row), device).item()))
            # Hard gate: full 6-way, neighbour-masked so only the current-state
            # field distinguishes the hypotheses (no previous/next leakage).
            masked_nll = {s: float(_nll(model, tokenizer, masked_state_row(row, s), device).item()) for s in STATE_ORDER}
            best_wrong = min(masked_nll[s] for s in negative_states(true))
            margins6.append(best_wrong - masked_nll[true])
            predicted = min(masked_nll, key=masked_nll.get)
            argmin_hits.append(predicted == true); argmin_confusion[true][predicted] += 1
            pid_margins.append((row["participant_id"], best_wrong - masked_nll[true]))
            # Draw the V30 counterfactual from the held-out participant pool,
            # not only the small generation-audit prefix.  Restricting it to
            # ``selected`` made coverage depend on file order and incorrectly
            # failed valid V30 audits when a local KC context appeared once in
            # the first N examples.
            value_negative = v30_counterfactual_row(row, rows, rng)
            if value_negative is not None:
                value_pairs.append((correct_nll, float(_nll(model, tokenizer, value_negative, device).item())))
            prefix = tokenizer.apply_chat_template([{"role":"user", "content":render_step_prompt(row)}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            ids = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = model.generate(input_ids=ids, do_sample=False, max_new_tokens=STEP_GENERATION_MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True)
            text = first_sentence(tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True))
            outputs.append({"sample_id": row["sample_id"], "participant_id": row["participant_id"], "current_state": row["current_state"], "target_text": row["target_text"], "generated_text": text, "q7_gold_used": False, "format_failure": step_format_failure(text), "template_risk": step_template_risk(text)})
    model.train()
    counts = Counter(item["generated_text"] for item in outputs)
    per_state = {state: max(Counter(item["generated_text"] for item in outputs if item["current_state"] == state).values(), default=0) / max(1, sum(item["current_state"] == state for item in outputs)) for state in STATE_ORDER if any(item["current_state"] == state for item in outputs)}
    increase = sum(wrong) / len(wrong) - sum(correct) / len(correct)
    mean_margin6 = sum(margins6) / len(margins6)
    argmin_accuracy = sum(argmin_hits) / len(argmin_hits)
    by_pid: dict[str, list[float]] = defaultdict(list)
    for pid, margin in pid_margins: by_pid[pid].append(margin)
    pid_means = [sum(values) / len(values) for values in by_pid.values()]
    boot_rng = random.Random(12345)
    boots = sorted(sum(boot_rng.choice(pid_means) for _ in pid_means) / len(pid_means) for _ in range(1000))
    margin_ci = [boots[24], boots[974]]
    value_increase = sum(wrong - correct for correct, wrong in value_pairs) / len(value_pairs) if value_pairs else None
    template_risk_rate = sum(item["template_risk"] for item in outputs) / len(outputs)
    gates = {"q7_gold_used": True, "state_6way_margin_positive": mean_margin6 > 0, "v30_counterfactual_coverage": (not require_v30) or len(value_pairs) / len(outputs) >= .9, "v30_permutation_nll_increase": (not require_v30) or (value_increase is not None and value_increase > 0), "exact_top_share": max(counts.values()) / len(outputs) < .25, "per_state_top_template": all(value < .40 for value in per_state.values()), "length_compliance": sum(not x["format_failure"] for x in outputs) / len(outputs) >= .80, "format_failure": not any(x["format_failure"] for x in outputs), "template_risk": template_risk_rate <= max_template_risk}
    return {"example_nll": sum(correct)/len(correct), "state_6way_mean_margin": mean_margin6, "state_6way_argmin_accuracy": argmin_accuracy, "state_6way_margin_participant_ci95": margin_ci, "state_6way_argmin_confusion": {t: dict(argmin_confusion[t]) for t in STATE_ORDER}, "state_permutation_nll_increase_single_negative": increase, "v30_permutation_nll_increase": value_increase, "v30_counterfactual_coverage": len(value_pairs) / len(outputs), "exact_top_share": max(counts.values())/len(outputs), "per_state_top_template_share": per_state, "length_compliance": sum(not x["format_failure"] for x in outputs)/len(outputs), "structural_failure_rate": sum(x["format_failure"] for x in outputs)/len(outputs), "template_risk_rate": template_risk_rate, "gate_status": "PASS" if all(gates.values()) else "FAIL", "checks": gates}, outputs


def train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_step_config(args.config); torch = require_training_runtime(require_cuda=True)["torch"]
    train_rows, validation_rows = read_jsonl(args.train_records), read_jsonl(args.validation_records)
    instrument_aux_rows = read_jsonl(args.instrument_aux_records) if args.instrument_aux_records else []
    coverage = validate_rows(train_rows, formal=args.formal); validate_rows(validation_rows, formal=args.formal)
    instrument_aux_coverage = validate_instrument_aux_rows(instrument_aux_rows) if instrument_aux_rows else Counter()
    if {r["participant_id"] for r in train_rows} & {r["participant_id"] for r in validation_rows}: raise StepTrainingError("participant split leakage")
    base = load_training_config(Path(args.config).parent / cfg["base_training_config"])
    # The legacy base config supplies runtime/model invariants only. V13.8 owns
    # its adapter shape so a Planner experiment cannot silently choose it.
    base = replace(base, raw={**base.raw, "lora": {**cfg["lora"], "bias": "none"}})
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tokenizer, model = load_tokenizer_and_causal_lm(base, model_path=args.model_path, adapter_name="step_realizer", qlora=args.qlora, device_index=args.device_index, trainable=True)
    model.train(); device = torch.device(f"cuda:{args.device_index}")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    rng = random.Random(args.seed); seen = Counter(); history=[]; best=None; args.output_dir.mkdir(parents=True, exist_ok=True)
    lambda_v30 = float(cfg["training"].get("lambda_v30", 0.0)); v30_margin = float(cfg["training"].get("v30_margin", 0.0))
    state_neg_cap = int(cfg["training"].get("state_negatives_per_update", 0))  # 0 => all five wrong states
    aux_fraction = float(cfg["training"].get("instrument_aux_fraction", 0.0))
    if aux_fraction and not instrument_aux_rows:
        raise StepTrainingError("instrument_aux_fraction requires --instrument-aux-records")
    main_indices = state_balanced_indices(train_rows, count=cfg["training"]["max_updates"], rng=rng)
    aux_indices = state_balanced_indices(instrument_aux_rows, count=cfg["training"]["max_updates"], rng=rng) if instrument_aux_rows else []
    for update, main_index in enumerate(main_indices, 1):
        use_aux = bool(instrument_aux_rows) and rng.random() < aux_fraction
        row = instrument_aux_rows[aux_indices[update - 1]] if use_aux else train_rows[main_index]
        seen[row["current_state"]]+=1
        true_state = row["current_state"]
        positive=_nll(model, tokenizer, row, device)  # real-neighbour CE objective (matches inference)
        pos_masked=_nll(model, tokenizer, masked_state_row(row, true_state), device)
        negs = negative_states(true_state)
        if state_neg_cap and state_neg_cap < len(negs):
            negs = rng.sample(negs, state_neg_cap)
        # Neighbour-masked hinge against every wrong state so gradient teaches the
        # full current-state -> text mapping, not a single confusable pair.
        hinge = positive.new_zeros(())
        for wrong_state in negs:
            hinge = hinge + torch.relu(cfg["training"]["margin"] + pos_masked - _nll(model, tokenizer, masked_state_row(row, wrong_state), device))
        hinge = hinge / max(1, len(negs))
        value_negative = v30_counterfactual_row(row, train_rows, rng) if not use_aux else None
        value_hinge = positive.new_zeros(())
        if lambda_v30 and value_negative is not None:
            value_hinge = torch.relu(v30_margin + positive - _nll(model, tokenizer, value_negative, device))
        loss=positive + cfg["training"]["lambda_state"]*hinge + lambda_v30*value_hinge
        loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), cfg["training"]["max_grad_norm"]); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        item={"update":update,"loss":float(loss.item()),"ce":float(positive.item()),"state_hinge":float(hinge.item()),"v30_hinge":float(value_hinge.item()),"n_state_negatives":len(negs),"source_role":"instrument_state_aux" if use_aux else "v7_personalized", "state_seen":dict(seen)}
        if update % cfg["training"]["evaluation_interval"] == 0 or update == cfg["training"]["max_updates"]:
            metrics, audit=_evaluate(torch, model, tokenizer, validation_rows, device, cfg["training"]["max_eval_examples"], require_v30=lambda_v30 > 0, max_template_risk=float(cfg["selection"].get("pilot_template_risk", 0.2))); item["validation"]=metrics; write_jsonl(args.output_dir / "generation_audit" / f"update_{update:04d}.jsonl", audit)
            if metrics["gate_status"] == "PASS" and (best is None or metrics["example_nll"] < best["example_nll"]):
                best={**metrics,"update":update}; model.save_pretrained(args.output_dir / "best_adapter")
        history.append(item)
    model.save_pretrained(args.output_dir / "last_adapter")
    write_jsonl(args.output_dir / "training_history.jsonl", history)
    manifest={"schema_version":"step_realizer_run_v1","q7_gold_used":False,"pi_used":False,"formal":args.formal,"status":"ENGINEERING_SMOKE_ONLY" if not args.formal else ("METHOD_PASS" if best else "METHOD_GATE_FAIL"),"state_coverage":dict(coverage),"six_state_coverage":all(coverage[s]>0 for s in STATE_ORDER),"instrument_aux_records":len(instrument_aux_rows),"instrument_aux_state_coverage":dict(instrument_aux_coverage),"instrument_aux_sha256":sha256_file(args.instrument_aux_records) if args.instrument_aux_records else None,"best":best,"base_config_sha256":base.sha256,"data_train_sha256":sha256_file(args.train_records),"data_validation_sha256":sha256_file(args.validation_records),"state_ontology_sha256":sha256_file(Path(__file__).resolve().parents[1] / "configs" / "state_ontology_v2.json"),"step_prompt_sha256":hashlib.sha256(render_step_prompt(train_rows[0]).encode()).hexdigest(),"base_model_sha256":directory_sha256(Path(args.model_path)),"config_sha256":sha256_file(args.config)}
    write_json(args.output_dir / "status.json", manifest); return manifest


def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True);p.add_argument("--train-records",type=Path,required=True);p.add_argument("--validation-records",type=Path,required=True);p.add_argument("--instrument-aux-records",type=Path);p.add_argument("--model-path",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device-index",type=int,default=7);p.add_argument("--seed",type=int,default=20260725);p.add_argument("--qlora",action="store_true");p.add_argument("--formal",action="store_true")
    args=p.parse_args(argv)
    try:
        print(json.dumps(train(args),ensure_ascii=False));return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "status.json", {
            "schema_version":"step_realizer_run_v1",
            "q7_gold_used":False,
            "pi_used":False,
            "formal":args.formal,
            "status":"ENGINEERING_SMOKE_FAIL" if not args.formal else "METHOD_GATE_FAIL",
            "failure_type":type(exc).__name__,
            "failure_message":str(exc),
        })
        raise
if __name__=="__main__": raise SystemExit(main())
