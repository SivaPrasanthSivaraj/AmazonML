# Full resumable pipeline

The production runner performs the complete sequence without notebook handoffs:

1. normalize all train/test records;
2. build full-pool classical indexes and S1-to-target candidates;
3. encode three pinned E5 views and run IVF ANN retrieval;
4. union routes in bounded shards and mine hard negatives;
5. compute the same leakage-safe lexical and direct-E5 features for every pair;
6. train, calibrate, evaluate, and refit LightGBM;
7. repeat retrieval/features for test and write both required TSV files.

Every expensive stage and shard is recorded in `pipeline_state.json`. Rerunning the
same command with the same working directory skips completed outputs.

## Hardware

The complete three-view run needs a CUDA GPU, at least 30 GB RAM, and at least
35 GB free working storage. A persistent AWS GPU instance with a 100 GB EBS volume
is safer than temporary Kaggle storage for this full run. Kaggle is suitable only
when its current session exposes at least 35 GB free disk.

## Install

```bash
python -m pip install -r requirements-production.txt
export PYTHONPATH="$PWD/src"
```

Kaggle already provides PyTorch. Do not replace its CUDA build with a CPU wheel.
If GPU FAISS is unavailable, `--faiss-device auto` safely uses CPU IVF search while
E5 encoding remains on CUDA.

## One command

```bash
python -u -m ber.pipeline \
  --dataset-dir /path/to/dataset \
  --work-dir /path/to/persistent/amazonml_full
```

No intermediate files need to be pasted into another notebook. If the process is
interrupted, run the exact command again. Final files are written to:

```text
<work-dir>/output/matching_results.tsv
<work-dir>/output/candidate_pairs.tsv
<work-dir>/output/submission_summary.json
```

Validate them with the organizer script before uploading:

```bash
python /path/to/student_resource/utils/validate_submission.py \
  --matching <work-dir>/output/matching_results.tsv \
  --candidate <work-dir>/output/candidate_pairs.tsv \
  --test-dir /path/to/dataset/test
```

## Important design constraints

- Retrieval is S1-to-S2/S3 for production, rather than the target-to-S1 direction
  used by the original small ANN ceiling experiment.
- All available truth links are forced into the fitting partition only. They are
  never injected into calibration or holdout candidates.
- ANN/classical ranks and route-presence flags may select candidates but never enter
  the model feature matrix.
- Name, address, and combined E5 cosines are computed directly for every retained
  positive and negative pair.
- Country is always an open string partition, so France is processed automatically.
