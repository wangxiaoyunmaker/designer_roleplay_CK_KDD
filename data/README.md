# Dataset Card

    Public dataset size: 163 anonymized designers.

    User IDs are deterministic release IDs (`D001`, `D002`, ...). The private mapping from
    release IDs to original participant IDs is not included in this public artifact.

    ## Included Files

    - `users.csv`: coarse participant metadata after removing raw IDs, timestamps, durations,
      devices, and free-text demographic "other" fields.
    - `scale_responses.csv`: 30-item trait scale responses with anonymized `user_id`.
    - `task_responses.csv`: raw task answers with anonymized `user_id` and public-safe columns.
    - `task_responses_expanded.csv`: expanded task answers where available.
    - `user_cluster_efa.csv`: cluster assignments.
    - `processed/`: lightweight extracted labels, SFT splits, factor scores, and CAPS artifacts.
    - `route_a_training_v1/formal/`: 162 eligible participants, 751 auxiliary
      choice-state records, 162 Q2 Planner trajectories, 972 cleaned Realizer
      records, participant-only five-fold assignments, manifests, and checksums.

    The Route A files use only `D001`-style IDs. They exclude Q7/Q8, raw identity
    fields, structured `V` states, option text, and card text.

    ## Privacy Notes

    This public release removes direct IDs and common re-identification fields. Before publishing,
    manually review free-form Chinese responses for names, institutions, companies, or other
    identifying details that automated pattern checks may not catch.
