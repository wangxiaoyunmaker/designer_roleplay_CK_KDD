# PKVC Gold-Label Audit

## Evidence and Provenance

The current Q7 machine-extracted label corpus contains **486 segments from 162 users**. The expert-gold audit subset contains **15 segments from 5 users**. These figures are reported only as aggregates; this document intentionally omits source user identifiers and response text.

The local audit artifact reports 14/15 dominant-class matches (**93.3%**) on the expert-gold subset. The legacy v8.6 extractor report records combined validation of **57/63 = 93.7% dominant accuracy**, combining the 15-segment expert-gold subset with a 48-segment Claude hand-label subset.

The 48-segment Claude hand-label subset is useful as a diagnostic validation set, but it is **not independent human inter-annotator-agreement (IAA) evidence**. Current evidence contains no reported blinded, independent human IAA. Claims about label reliability should therefore be limited to extractor agreement with the available reference subsets, rather than claiming established human annotation reliability.

## What the Metrics Measure

Dominant accuracy evaluates whether the prediction identifies the first/head or dominant PKVC class. Sequence LCS-F1 evaluates ordered overlap between multi-state sequences. The two measures are not contradictory: a prediction can identify the main class correctly while compressing or deduplicating intermediate states, producing high dominant accuracy but weaker sequence LCS-F1.

This distinction matters for downstream use. Dominant-class analyses have the strongest available validation support; analyses that depend on state ordering or intermediate transitions should retain and report sequence-level uncertainty. The audit artifact additionally records sequence edit and set-overlap diagnostics for the 15 expert-gold segments, but it does not supply independent human IAA or a sequence LCS-F1 estimate for the combined 63-segment validation.

## Extractor/Backbone Dependence Risk

If GPT-4o is both the inference backbone and the extractor used to label GPT-4o outputs, extractor bias is plausible. Shared modeling assumptions can inflate apparent agreement or preserve the same systematic errors in generated outputs and extracted labels.

Before treating the labels as model-independent evidence, add either a human-only validation subset or a robustness check using a non-GPT-4o extractor. The preferred safeguard is a human-only subset because it directly tests the target construct rather than agreement between model pipelines.

## Minimal Human-Only Validation

Run a blinded annotation pilot on **30 random Q7 segments**, stratified jointly by `seg1`/`seg2`/`seg3` and PKVC class. Use **two annotators**, with neither annotator shown model predictions or extractor output.

Report:

- Cohen's kappa for the dominant PKVC class.
- LCS-F1 and sequence agreement for the ordered state sequence.
- The sampling and stratification procedure, annotation instructions, adjudication policy, and any exclusions.

This pilot establishes the missing human-agreement evidence and tests whether the observed dominant accuracy survives outside the GPT-4o extraction pathway.

## Interpretation Boundary

The v8.6 57/63 result supports a narrow statement: under the recorded local validation protocol, dominant-class agreement was 93.7%. It does not establish independent human IAA, unbiased extraction, or equivalent fidelity for ordered state sequences. Those limits should accompany any use of these labels in technical claims.
