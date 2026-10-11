# Metrics Audit: Sequence Edit Distance and NL Specificity

## Scope

This audit compares the paper text extracted in `tmp/pdfs/17317_raw.txt`
with the public metric reference and the two current Python implementations.
It records the discrepancy only; no metric script is changed by this task.

## Finding: macro-TED Names and Units Do Not Agree

The current sources describe different metrics under the `macro-TED` label:

| Source | Current name or claim | Actual unit / computation |
| --- | --- | --- |
| Paper, Section 5 | `Macro Tree Edit Distance (macro-TED)` | Calls the measure class-balanced normalized Levenshtein distance. |
| Paper, Appendix A.1 | `Macro Tree Edit Distance (macro-TED)` | Defines Levenshtein divided by `max(|p|, |g|, 1)`, hence a value in `[0, 1]`. |
| `scripts/final_eval_with_bootstrap.py` | `seg2_macro_TED` | Computes raw (unnormalized) Levenshtein counts, then averages those counts by ground-truth first-state class. |
| `sft_pipeline/eval/metrics.py` | `factor_distance` / `factor_hamming` alias | Computes Levenshtein divided by `max(len(pred), len(truth))`, hence a normalized sequence distance in `[0, 1]`. |

The paper's result tables independently corroborate the evaluator's raw-count
behavior: reported macro-TED values include `2.489`, `2.121`, `2.804`, and
`1.821`. A normalized distance cannot exceed `1`. Therefore, the currently
reported results are raw class-balanced sequence edit distances, not the
normalized quantity defined in Appendix A.1.

`Tree Edit Distance` is also inaccurate terminology here. All implementations
operate on linear PKVC sequences with Levenshtein insertion, deletion, and
substitution operations; none represents or compares trees.

## Recommendation

For the existing reported tables, preserve the underlying values and rename the
metric to **macro sequence edit distance**. Update the paper body, tables,
captions, and appendix to define it as class-balanced, unnormalized
Levenshtein distance. Remove every `[0,1]` claim associated with this reported
metric. This is the least disruptive correction because the published values
are already demonstrably unnormalized.

For a future recomputation, the preferred comparable metric is **macro
normalized sequence edit distance**: divide each per-designer Levenshtein
distance by `max(|p|, |g|, 1)` before the class-balanced macro average, and
standardize every evaluator on that definition. Recomputed tables must replace
the existing raw-count values before this normalized name or `[0,1]` range is
used.

Do not silently replace the current evaluator with normalization while retaining
the existing tables. That would make the code, documentation, and reported
numbers disagree again.

## NL Specificity Placement

NL specificity is already defined in
`release/public_repo/docs/briefs/metric_reference.md` as a three-segment
personalization metric, including its cosine-ratio formula, embedding model,
and `m_nl_specificity` implementation. The current paper's main summary table
does not list it. The paper should either report NL specificity in the main
table (and include its results) or explicitly state that it is supplementary,
with a pointer to the appendix or supplementary material that reports it. The
choice should be reflected consistently in the narrative, table count, and
metric reference.

## Follow-up Checklist

1. Decide whether the revision preserves existing raw-count results or reruns
   every table with normalized distances.
2. Apply the matching terminology and formula consistently in the paper,
   `metric_reference.md`, and evaluator output labels.
3. Add a regression test that distinguishes raw from normalized distance on a
   sequence pair such as `["V"]` versus `["P", "K", "C"]`.
4. Resolve NL specificity's main-table versus supplementary status and make the
   result location explicit.
