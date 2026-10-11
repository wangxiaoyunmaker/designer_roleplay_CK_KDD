# V30-Conditioned KC Operational Ontology V1

## 1. Scope and Intended Claim

This pipeline mines profile-conditioned operational routing rules from the Q2
trajectories. Its state space is fixed to the following six KC states:

\[
\mathcal S=\{K,C,K\!\to\!C,C\!\to\!K,K\!\to\!K,C\!\to\!C\}.
\]

`START` and `STOP` are boundary tokens rather than cognitive states. The
extractor fails closed if a planner record is not Q2 or contains a label outside
the six-state ontology. It does not read Q7 prompts, outputs, or gold labels
during discovery, model selection, rule filtering, or prompt compilation.

The intended result is an executable routing policy of the form

\[
r=(\text{profile condition},z_t)
\Rightarrow
(z_{t+1},\text{semantic operation},\text{guard}),
\]

not a descriptive statement such as "this subgroup has a higher state
frequency." The resulting JSON can be deterministically rendered as a natural
language block and inserted into the Q7 reasoning prompt.

## 2. Inputs

| File | Required fields | Use |
| --- | --- | --- |
| `participants.jsonl` | `participant_id`, 30-element `v30` | Profile conditions |
| `planner_records.jsonl` | `participant_id`, `question_id=Q2`, `kc_states` | Complete Q2 trajectories |
| `trait_items_30.csv` | ordered trait ID, name, positive and negative poles | Human-readable triggers |

Each Q2 trajectory is expanded as

\[
\text{START}\to z_1\to\cdots\to z_T\to\text{STOP}.
\]

Repeated states and variable lengths are preserved. Normalized position is
computed as \(p_t=t/T\).

## 3. Discovery Model

For every observed source state \(a\), the extractor fits an L1-regularized
one-vs-rest next-state model. For target state \(b\),

\[
\operatorname{logit}P(z_{t+1}=b\mid z_t=a)
=\alpha_{ab}+\boldsymbol\beta_{ab}^{\top}\widetilde{\mathbf v}_i
+\gamma_{1,ab}(2p_t-1)+\gamma_{2,ab}(2p_t-1)^2.
\]

V30 scores are standardized for discovery. Regularization strength is selected
by participant-grouped cross-validation, so transitions from one participant
cannot occur in both the training and validation partitions.

For each `(source state, target state, trait)` candidate, participant-level
bootstrap resampling estimates:

- selection frequency;
- coefficient sign consistency;
- percentile coefficient interval;
- transition and participant support.

The default discovery-stability gates follow the established extraction
protocol:

- supporting participants \(\ge 20\);
- bootstrap selection frequency \(\ge 0.90\);
- sign consistency \(\ge 0.95\);
- standardized \(|\beta|\ge 0.15\).

Passing these gates produces a `stable_exploratory` rule, not a validated rule.

## 4. Participant-Held-Out Validation

Every stable candidate receives two additional tests.

### 4.1 Incremental held-out likelihood

Within each participant-held-out fold, the baseline predicts a binary target
transition using sequence position only. The augmented model adds one V30
dimension. The rule-level incremental loss is

\[
\Delta\mathcal L_r
=\mathcal L_{\text{position}+v_k}
-\mathcal L_{\text{position}}.
\]

Negative values indicate that the profile dimension improves unseen-participant
prediction. Standardization is recomputed using the training participants in
each fold.

### 4.2 Participant-level permutation and FDR

Out-of-fold baseline residuals are averaged within participant. The association
between each V30 dimension and the participant residual is compared with a null
distribution produced by participant-level permutation. Benjamini-Hochberg
correction is applied jointly to all source-target-trait candidates.

A rule receives `rule_status=validated` only when it passes all discovery gates
and additionally satisfies:

- BH-FDR \(q\le0.10\);
- mean held-out \(\Delta\mathcal L_r<0\);
- at least 60% of held-out folds improve;
- held-out coefficient direction agrees with the full-data direction in at
  least 60% of folds.

The procedure permits zero validated rules. This is necessary for the rule
library to distinguish lack of evidence from extraction failure.

## 5. Operational Rule Compilation

A positive coefficient is oriented toward the high-score trait pole; a negative
coefficient is oriented toward the low-score pole. By default, scores \(\ge4\)
activate the positive pole and scores \(\le3\) activate the negative pole.

The statistically selected target state is compiled into a semantic operation:

| Next state | Executable semantic operation |
| --- | --- |
| `K` | Add verifiable facts, experience, data, or constraints |
| `C` | Form a concrete concept, principle, or candidate solution |
| `KtoC` | Derive a new solution from evidence or constraints |
| `CtoK` | Validate an existing solution against evidence, user impact, or feasibility |
| `KtoK` | Derive consequences, limitations, or relations among known evidence |
| `CtoC` | Refine, combine, replace, or adjust an existing solution |

This semantic operation is a deterministic realization of the KC target state;
it is not presented as an independently learned psychological variable. Each
operation also carries a guard, such as blocking `STOP` until a required
validation or alternative has been articulated.

Example JSON:

```json
{
  "rule_id": "KCOR_example",
  "rule_status": "validated",
  "trigger": {
    "profile": {
      "trait_id": "T04",
      "operator": "score_lte",
      "threshold": 3,
      "stance": "技术审慎主义"
    },
    "current_state": "C",
    "context_policy": "activate_only_when_semantically_relevant"
  },
  "action": {
    "next_state": "CtoK",
    "routing": "prefer",
    "semantic_operation": "检验已有方案所依赖的证据、用户影响和可行性",
    "guard": "完成至少一项证据或风险检查前不得结束推理。"
  }
}
```

## 6. Prompt Injection

At inference time, the renderer receives a participant's V30 profile and the
frozen rule library. A rule is activated only if:

1. its evidence status is allowed by the run configuration;
2. the participant satisfies its profile threshold;
3. the current reasoning state matches the rule trigger;
4. the semantic operation is relevant to the public scenario.

Rules are soft routing preferences, not forced facts. The prompt explicitly
forbids inventing evidence merely to satisfy a rule. The confirmatory setting
injects `validated` rules only. Injection of `stable_exploratory` rules must be
reported as an ablation.

The required Q7 comparison is:

\[
\text{correct-profile rules}
\quad\text{vs}\quad
\text{shuffled-profile rules}
\quad\text{vs}\quad
\text{no rules}.
\]

This comparison tests whether the routing effect is profile-specific rather
than a generic prompt-length or diversity effect.

## 7. Reproducible Commands

Formal single-dimension and double-dimension extraction share resampling
protocol `participant_resampling_v1`: participant-level bootstrap >= 1,000,
participant-level permutations >= 1,000, and exactly five participant-held-out
folds. The frozen runtime is Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, and
scikit-learn 1.8.0. Formal extraction fails closed under a different runtime,
because solver-version drift changes held-out fold decisions. Audit artifacts
must carry the same protocol ID, software versions, and input hashes before
their rule counts are combined.

Formal extraction:

```bash
python tools/kc_operational_ontology.py mine \
  --participants data/route_a_training_v1/formal/participants.jsonl \
  --planner data/route_a_training_v1/formal/planner_records.jsonl \
  --trait-items artifacts/reproducibility/trait_items_30.csv \
  --output-dir artifacts/kc_operational_ontology_v1_uniform \
  --run-kind formal \
  --bootstrap 1000 \
  --permutations 1000 \
  --folds 5
```

The frozen uniform run produced 12 `stable_exploratory` single-dimension rules
and zero `validated` rules.

`--run-kind formal` fails unless both resampling counts are at least 1,000.
Development runs cannot emit the `validated` status even when their provisional
statistics cross the numerical gates.

Render a participant-specific block:

```bash
python tools/kc_operational_ontology.py render \
  --rules artifacts/kc_operational_ontology_v1/rules.json \
  --participants data/route_a_training_v1/formal/participants.jsonl \
  --participant-id D001 \
  --scenario-file q7_public_scenario.txt \
  --output outputs/D001_kc_rules_prompt.md
```

Exploratory rules require explicit authorization:

```bash
python tools/kc_operational_ontology.py render \
  --rules artifacts/kc_operational_ontology_v1/rules.json \
  --participants data/route_a_training_v1/formal/participants.jsonl \
  --participant-id D001 \
  --status validated stable_exploratory
```

## 8. Outputs and Auditability

| Artifact | Content |
| --- | --- |
| `candidates.jsonl` | All fitted candidates and all validation statistics |
| `rules.json` | Stable operational rules and injection schema |
| `audit.json` | Counts, hyperparameters, input hashes, state contract, and Q7 non-use declaration |

Artifacts contain aggregate evidence only. Participant identifiers are used to
group folds and bootstrap samples but are never written to the rule library.

## 9. Limitation

Q2 provides one trajectory per participant in one shared scenario. It can
support profile-conditioned procedural preferences, but it cannot identify a
scenario-specific trigger by itself. The runtime semantic-relevance condition
therefore acts as an applicability guard rather than a learned context effect.
Claims of cross-scenario personalization require the Q7 correct/shuffled/no-rule
test or additional repeated trajectories collected under multiple scenarios.
