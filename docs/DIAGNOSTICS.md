# Diagnostic Annotation Schemas

The paper includes three diagnostic analyses that rely on human annotations. The summary script accepts JSONL files with the following fields.

## Decomposition Quality

Path:

```text
data/diagnostics/decomposition_quality.jsonl
```

Required fields:

- `item_id`
- `question`
- `triplet`
- `valid`

Optional agreement fields:

- `annotator_a_valid`
- `annotator_b_valid`

## Holistic-Atomic Gap

Path:

```text
data/diagnostics/holistic_atomic_gap.jsonl
```

Required fields:

- `trial_id`
- `question`
- `holistic_correct`
- `atomic_correct`

The summary reports holistic accuracy, atomic verification accuracy, and their gap.

## Failure Taxonomy

Path:

```text
data/diagnostics/failure_taxonomy.jsonl
```

Required fields:

- `failure_id`
- `benchmark`
- `question`
- `category`

Optional agreement fields:

- `annotator_a_category`
- `annotator_b_category`

Run:

```bash
bash scripts/run_diagnostics_summary.sh
```
