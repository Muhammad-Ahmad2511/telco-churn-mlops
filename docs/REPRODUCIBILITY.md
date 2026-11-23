# Reproducibility and data versioning

This document covers two requirements from spec §3.2:

1. **Retrieving an earlier code version together with its dataset**, and showing that it reproduces the same model.
2. **Handling merge conflicts**, including conflicts in `.dvc` and `dvc.lock` files.

Code is versioned in Git. Data and pipeline outputs are versioned with DVC, using DagsHub as the remote. Every MLflow run and every registered model version records both versions:

| Tag | Meaning |
|---|---|
| `git_commit` | Exact code version used for training |
| `git_dirty` | `False` means it was trained from committed code (`register.py` refuses dirty runs) |
| `dvc_data_md5` | md5 of the raw dataset, taken from `data/raw/telco_churn.csv.dvc` |

Starting from any registered model, these tags give you the commit to check out and the data version to pull.

---

## 1. Demo: recover model v1 from its code and data versions

**Target:** registered model `telco-churn-classifier` **version 1** (the Logistic Regression baseline).
Its registry tags are `git_commit = 6b6db68d8…` and `dvc_data_md5 = 0f9de68e012bd3aed5fa7cdc9fc421af`.
Its recorded test metrics are accuracy 0.737, precision 0.503, recall 0.783, F1 0.613 and ROC-AUC 0.842.

The demo was run on 2026-11-23 on Windows 11 in PowerShell, from the repo root with the project venv active. Everything below is real output.

### Step A: check out the old code

```powershell
git checkout 6b6db68
git log --oneline -1
```
```
HEAD is now at 6b6db68 feat: add baseline training, evaluation gate and MLflow tracking
6b6db68 (HEAD, origin/feat/data-validation, feat/data-validation) feat: add baseline training, evaluation gate and MLflow tracking
```

### Step B: delete the dataset and restore the version that belongs to this commit

```powershell
Remove-Item data\raw\telco_churn.csv
.venv\Scripts\dvc pull data/raw/telco_churn.csv.dvc
(Get-FileHash data\raw\telco_churn.csv -Algorithm MD5).Hash
Get-Content data\raw\telco_churn.csv.dvc
```
```
A       data\raw\telco_churn.csv
1 file added
0F9DE68E012BD3AED5FA7CDC9FC421AF
outs:
- md5: 0f9de68e012bd3aed5fa7cdc9fc421af
  size: 977501
  hash: md5
  path: telco_churn.csv
```

The restored file's hash matches the `.dvc` pointer in that commit and the `dvc_data_md5` tag of model v1. PowerShell prints hashes in upper case, but the values are identical.

> **Note:** in this old commit, the root `.gitignore` contained `data/raw/*`, which hid the `.dvc` pointer from DVC. A plain `dvc pull` silently did nothing, so the pointer file has to be named explicitly. The bug was fixed in PR #4 (`data/raw/*.csv`). On current commits, a plain `dvc pull` or `dvc checkout` works.

### Step C: retrain the old code on the old data

A temporary local MLflow store is used so the shared DagsHub experiment doesn't get an extra run.

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///repro-check.db"
.venv\Scripts\python -m src.models.train
Remove-Item Env:MLFLOW_TRACKING_URI
```
```
INFO accuracy = 0.7374
INFO precision = 0.5034
INFO recall = 0.7834
INFO f1 = 0.6130
INFO roc_auc = 0.8418
INFO Quality gate: PASSED []
```

| Metric | Registered v1 | Retrained from `6b6db68` + data `0f9de68e…` |
|---|---|---|
| Accuracy | 0.737 | 0.7374 |
| Precision | 0.503 | 0.5034 |
| Recall | 0.783 | 0.7834 |
| F1 | 0.613 | 0.6130 |
| ROC-AUC | 0.842 | 0.8418 |

**The result is identical.** This is because the split and the model both use the fixed `random_state: 42` from `configs/`, the dependencies are pinned in `requirements.txt`, and the data is fixed by its md5.

### Step D: return to the current version

```powershell
git checkout main
.venv\Scripts\dvc checkout
Remove-Item repro-check.db
Remove-Item -Recurse mlruns
git status
```
```
Switched to branch 'main'
nothing to commit, working tree clean
```

### Current pipeline: one command

Since PR #8, the whole training path is a DVC pipeline (`dvc.yaml`: validate → preprocess → tune). On any commit that has `dvc.yaml`:

```powershell
.venv\Scripts\dvc pull     # data + processed outputs recorded in dvc.lock
.venv\Scripts\dvc repro    # re-runs only stages whose deps/params changed
.venv\Scripts\dvc metrics show
```

`dvc.lock` records the md5 of every stage's inputs, parameters and outputs, so the same commit always describes the same data, configuration and results. Running `dvc repro` a second time without changes skips every stage.

---

## 2. Merge conflicts, including `.dvc` and `dvc.lock`

### Why `.dvc` files conflict

A `.dvc` file is a small text pointer. If two branches each change the same dataset, for example both run `dvc add` on `data/raw/telco_churn.csv`, both branches edit the same `md5:` and `size:` lines, and `git merge` reports a conflict:

```
<<<<<<< HEAD
- md5: 0f9de68e012bd3aed5fa7cdc9fc421af
  size: 977501
=======
- md5: <md5 of the other branch's version>
  size: <size of the other branch's version>
>>>>>>> feature-branch
  hash: md5
  path: telco_churn.csv
```

The data itself never conflicts in Git, because only the pointer is in Git. **The team has to decide which data version is correct.** Git can't merge two datasets line by line.

### How we resolve a conflict in a `.dvc` file

1. **Decide which data version wins.** Discuss it in the PR. If both changes are needed, for example new rows on each branch, build the combined dataset deliberately instead of choosing one side.
2. **Resolve the pointer:**
   - To keep one side: `git checkout --ours data/raw/telco_churn.csv.dvc` (current branch) or `git checkout --theirs data/raw/telco_churn.csv.dvc` (incoming branch). You can also edit the file so that exactly one `md5`/`size` pair remains.
   - If the data was combined: run `dvc add data/raw/telco_churn.csv` to write a fresh pointer for the new file.
3. **Make the workspace match the pointer:** `dvc checkout data/raw/telco_churn.csv.dvc`. If that version isn't in the local cache, run `dvc pull data/raw/telco_churn.csv.dvc`.
4. **Validate before committing:** `dvc repro validate`, or `python -m src.data.validation`.
5. **Commit and upload:** `git add data/raw/telco_churn.csv.dvc`, then `git commit`, then `dvc push`. Without `dvc push`, teammates and CI get a pointer to data that isn't on DagsHub.

### How we resolve a conflict in `dvc.lock`

`dvc.lock` is generated by DVC, so **never resolve it by hand**:

1. Resolve the real inputs first: code, `configs/*.yaml`, and any `.dvc` pointers.
2. Take either side of `dvc.lock`, for example `git checkout --theirs dvc.lock`.
3. Run `dvc repro`. DVC re-runs the affected stages and rewrites `dvc.lock` with the correct hashes.
4. Run `git add dvc.lock reports/`, then commit, then `dvc push`.

### Conflicts in params and metrics

- `configs/*.yaml` are normal text files, so resolve them as usual. Then run `dvc repro`, because changed params invalidate the stages that use them.
- `reports/metrics.json` and the other reports are pipeline outputs. Don't merge them by hand; regenerate them with `dvc repro`.

### How we avoid conflicts

- Only one branch at a time changes a given dataset. Data changes go in their own PR (`data:` commit prefix).
- Branch protection on `main` requires a PR and green CI (`Lint and test`, `Security scan`, `Docker build, scan and smoke test`). CI runs `dvc pull`, so a pointer to data that was never pushed fails the build.
- Always run `dvc push` together with `git push` whenever a `.dvc` file or `dvc.lock` changes.

---

## Evidence

Screenshots of steps A–D and of the DagsHub registry page for version 1 (showing the `git_commit` and `dvc_data_md5` tags) are kept with the team's project evidence.
