# Artifact Inventory

This matrix records appendix and supplement readiness from the reproducibility
artifacts and the intended public release. A listed public path is evidence of
availability, not a privacy or scientific-validity approval.

## Status Categories

| Status | Meaning |
| --- | --- |
| `PUBLIC_READY` | Present in the public release and suitable for release subject to the repository's stated privacy review. |
| `PRIVATE_ONLY` | Present only in the restricted data pack or otherwise requires authorized access. |
| `DERIVABLE_NOW` | Can be regenerated from the available code and documented inputs, subject to the stated runtime dependencies. |
| `NEEDS_HUMAN_WORK` | Requires human review, curation, validation, or documentation before it can support a release claim. |
| `MISSING_OR_UNFOUND` | No complete artifact was found at the inspected public or private release locations. |
| `DO_NOT_PUBLISH_RAW` | Must not be released in raw form because of identifying or otherwise unsafe content. |

## Readiness Matrix

| Required artifact | Status | Evidence / location | Release action or limitation |
| --- | --- | --- | --- |
| 30 trait items, polarity, and scoring | `PUBLIC_READY` | `docs/notes/PKVC_30节点本体.md` defines all 30 items, plus/minus poles, and the 1-6 to `-`/`0`/`+` scoring rule. | Cross-check the appendix against this locked v1.0 source before release. |
| Dimension-name rationale | `NEEDS_HUMAN_WORK` | The ontology identifies names as a locked 30-item mapping, but its rationale is limited to project-history notes. | Write and review a dimension-by-dimension rationale; do not imply independent psychometric validation from the existing notes. |
| 163-anonymized trait-response matrix | `PUBLIC_READY` | `artifacts/reproducibility/trait_response_matrix_30d.csv` has 163 data rows, 30 trait columns, anonymized `D###` IDs, and `has_q7_gold`. The public release also includes `data/scale_responses.csv`. | Retain the dataset card's required manual review of free-text materials; this matrix itself contains numeric scores only. |
| Q1-Q7 full task text | `NEEDS_HUMAN_WORK` | Q1 and Q3-Q6 stems are in `sft_pipeline/data/prep.py`; Q2 is documented in `docs/training_data_formats/Q2_cardid_ranking.md`; Q7 background and segment tasks are in `scripts/run_3_baselines_q7.py`. | Consolidate and human-review one version-controlled questionnaire appendix. Current texts are split across implementation and training documents, with noted historical wording drift. |
| 36 procedures full table | `PUBLIC_READY` | `artifacts/reproducibility/procedures_36.csv` has 36 data rows and includes IDs, conditions, coefficients, predicates, outcomes, and natural-language descriptions. | Publish as a generated supplement table with its generator/version provenance. |
| Per-participant selected 4-5 rules | `NEEDS_HUMAN_WORK` | `artifacts/reproducibility/activated_rules_by_designer.csv` has 644 selected active rows for 163 anonymized designers. The current data contain 155 designers with four rows and eight with three, not four to five for every participant. | Reconcile the stated 4-5 rule requirement with the active-selection policy before publication; do not claim complete 4-5 coverage. |
| Prompt versions B0-B4, ablation, and SFT prompt | `NEEDS_HUMAN_WORK` | B0-B4 conditions are documented in `sft_pipeline/config/config_v2.yaml`; prompt and ablation materials exist in `docs/briefs/` and `sft_pipeline/config/prompts/`; SFT formatting is documented in `docs/briefs/sft_data_format_v1.md`. | Freeze and export the exact prompts used for every reported run, with model/version/temperature metadata. Existing materials describe multiple evolving prompt families. |
| Q7 extractor labels | `NEEDS_HUMAN_WORK` | `data/processed/q7_extracted_v8.7_full.jsonl` contains labels produced by the v8.7 extractor. The extractor configuration and scripts identify GPT-4o use. | Label these as model-assisted extraction outputs, not human gold; perform privacy review of embedded response text before public release. |
| Expert-gold subset, original JSON filenames | `DO_NOT_PUBLISH_RAW` | `scripts/parse_expert_gold.py` expects `data/processed/expert_gold/*.json` and reads original per-user records. The task constraint states those filenames include personal identifiers. | Do not publish original expert-gold JSON files or filenames. Publish only a deidentified, human-reviewed derivative if authorized. |
| Expert-gold subset, deidentified evaluation derivative | `NEEDS_HUMAN_WORK` | `artifacts/reproducibility/gold_label_audit.json` provides a 15-segment audit summary, but no releasable expert-gold source subset was found. | Create a deidentified, provenance-preserving subset only after human confirmation of consent, identifiers, and gold-label provenance. |
| Hand-label subset | `NEEDS_HUMAN_WORK` | `data/processed/legacy/hand_labels.json` declares `labeler: claude` in its metadata. | Treat this as model-assisted labels, not human gold; rename/document it accordingly and obtain human annotation if gold labels are needed. |
| Metric scripts | `DERIVABLE_NOW` | Public scripts include `eval_vs_gold.py`, `eval_v8_6_vs_hand_gold.py`, `role_play_metrics.py`, `seg1_seg3_nl_metrics.py`, and `aggregate_cross_model_11metrics.py`; metric definitions are in `docs/briefs/metric_reference.md`. | Regeneration requires the documented input artifacts and, for extraction/inference scripts, configured API credentials and the named model endpoints. |
| Metric report outputs | `MISSING_OR_UNFOUND` | The public README states generated reports are intentionally excluded; no public `reports/` directory was found. The restricted pack contains only `processed_data_v3/q1q6_expansion_report.md`, not a complete metric-report bundle. | Regenerate and archive the exact report outputs, commands, environment/model metadata, and checksums before claiming reproductions. |

## Notes

- The repository dataset card describes 163 anonymized designers and requires manual
  review of free-form Chinese responses for residual identifiers.
- Model-generated labels, including GPT-4o extraction outputs and the
  Claude-labelled legacy subset, are not human gold labels.
- This inventory does not override the restricted-pack policy in
  `release/private_data_pack/README_RESTRICTED.md`.
