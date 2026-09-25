# Business Entity Resolution

Reproducible code for the Amazon ML Challenge 2026. The implementation is being
built in measured stages; no external business-identity data or lookup service is
used.

## Current commands

Run the streaming structural audit from `student_resource/`:

```bash
PYTHONPATH=../code/business_entity_resolution/src python -m ber.audit --dataset-dir dataset
```

Measure the deterministic validation split and exact normalized retrieval baseline:

```bash
PYTHONPATH=../code/business_entity_resolution/src python -m ber.baseline --dataset-dir dataset
```

Evaluate rare name/address tokens and long address-number blocking keys:

```bash
PYTHONPATH=../code/business_entity_resolution/src python -m ber.token_retrieval --dataset-dir dataset --summary-only
```

Evaluate overlapping name character 4-grams on a deterministic 1% S1 subset
against the complete target pools:

```bash
PYTHONPATH=../code/business_entity_resolution/src python -m ber.qgram_retrieval --dataset-dir dataset
```

Materialize, deduplicate, and rank the combined candidate routes on a deterministic
0.1% S1 subset:

```bash
PYTHONPATH=../code/business_entity_resolution/src python -m ber.candidate_ranking --dataset-dir dataset
```

Run unit tests from `code/business_entity_resolution/`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell equivalents:

```powershell
$env:PYTHONPATH = "..\code\business_entity_resolution\src"
python -m ber.audit --dataset-dir dataset

python -m ber.baseline --dataset-dir dataset

python -m ber.token_retrieval --dataset-dir dataset --summary-only

python -m ber.qgram_retrieval --dataset-dir dataset

python -m ber.candidate_ranking --dataset-dir dataset

Set-Location ..\code\business_entity_resolution
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

End-to-end blocking, model training, inference, and output-generation commands will
be added after their validation measurements are established.

## Measured validation results

The deterministic holdout contains 220,745 S1 entities and 764,320 true S1-to-S2/S3
links. Results below were measured on that holdout; they are not leaderboard scores.

| Candidate routes | S2 recall | S3 recall | Combined recall |
| --- | ---: | ---: | ---: |
| Exact normalized name/address | 35.86% | 31.58% | 33.65% |
| Exact + rare tokens/numbers (S1 DF <= 100) | 80.18% | 78.26% | 79.19% |
| Above + 4-character word prefixes/suffixes | 80.57% | 78.69% | 79.60% |

At the DF <= 100 cutoff, rare-token candidate occurrences total about 42.0 million
across both target sources for the holdout. This is an upper bound because a pair
sharing multiple blocking keys is counted multiple times. The next retrieval stage
must add typo-tolerant character evidence, then rank and deduplicate candidates to a
smaller per-target set.

The prefix/suffix experiment added only 3,163 of 764,320 holdout links (+0.41
percentage points) while raising the candidate-occurrence upper bound from about
42.0M to 71.4M. It is therefore rejected from the planned production generator;
internal character n-grams will be evaluated instead.

An overlapping name 4-gram experiment was then run on a deterministic 1% subset
(22,139 S1 entities, 76,382 true links) against the complete target pools. The
existing exact+token baseline reached 79.03% on this subset. Name 4-grams raised it
to 82.47% at S1 DF <= 50 (+3.44 points; 2.50M q-gram candidate occurrences) and
84.22% at DF <= 100 (+5.20 points; 7.11M occurrences). DF <= 50 is the provisional
efficiency setting; DF <= 100 remains useful for measuring the recall ceiling.

Candidate pairs were then materialized for a deterministic 0.1% S1 subset, with
sampled true targets ranked against the complete 2.21M-record S1 universe. Targets
had 66–68 candidates on average before address q-grams were added.

Overlapping 5-character address grams produced the largest blocking gain. On the
1% evaluation subset, exact + token/number + name/address q-grams reached 93.62%
combined recall at S1 DF <= 50 (versus 79.03% before q-grams). On the corrected
all-S1 ranking experiment, raw candidates averaged 139 for S2 and 148 for S3. A
lightweight RapidFuzz name/address reranker retained 94.09% S2 recall at top-20
versus a 94.28% ceiling, and 93.59% S3 recall versus a 93.93% ceiling. Top-20 is the
provisional pre-classifier cap.
