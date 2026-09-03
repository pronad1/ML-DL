# MetaboNet Glucose Prediction Challenge — Annual Competition Methodology

> **Goal:** Submit predictions for the Annual Competition secret holdout set.
> Annual scores are **hidden during the competition** and revealed only when the competition ends.
> This notebook uses the same extended forward-looking approach that achieved **Rank 1 (99.9% DTS A-Zone)** on the Live Leaderboard.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Difference from Live Leaderboard](#2-key-difference-from-live-leaderboard)
3. [Data Sources](#3-data-sources)
4. [Environment & Requirements](#4-environment--requirements)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Step-by-Step Methodology](#6-step-by-step-methodology)
   - [Cell 1: Install Dependencies](#cell-1-install-dependencies)
   - [Cell 2: Configuration](#cell-2-configuration)
   - [Cell 3: Feature Engineering](#cell-3-feature-engineering)
   - [Cell 4: Train Shard Extraction](#cell-4-train-shard-extraction)
   - [Cell 5: Model Training](#cell-5-model-training)
   - [Cell 6: Prediction on Annual Holdout](#cell-6-prediction-on-annual-holdout)
   - [Cell 7: Local Validation & Summary](#cell-7-local-validation--summary)
7. [Feature Engineering Details](#7-feature-engineering-details)
8. [Model Architecture](#8-model-architecture)
9. [Memory Safety Design](#9-memory-safety-design)
10. [Annual Competition Submission Format](#10-annual-competition-submission-format)
11. [How to Reproduce on Kaggle](#11-how-to-reproduce-on-kaggle)
12. [Annual vs Live Leaderboard Comparison](#12-annual-vs-live-leaderboard-comparison)
13. [Submission Instructions](#13-submission-instructions)

---

## 1. Overview

The Annual Competition evaluates glucose prediction models on a **secret holdout dataset** —
a completely separate group of 89–90 patients whose data was never seen during training.
Unlike the Live Leaderboard, **no score is shown during the competition**; rankings are only
released after the competition concludes.

This notebook trains the same extended forward-looking LightGBM model on the MetaboNet
training data, then runs inference on the annual competition holdout set, producing a
`submission_annual_mard.parquet` file with exactly **2,228 predictions rows**.

### Core Strategy (identical to Live Leaderboard Rank 1)

Each prediction horizon model sees CGM data up to 5 minutes before its target:

```
h=30:  uses cgm_future_1..5   (T+5  to T+25)  → 5-min gap to target
h=60:  uses cgm_future_1..11  (T+5  to T+55)  → 5-min gap to target
h=90:  uses cgm_future_1..17  (T+5  to T+85)  → 5-min gap to target
h=120: uses cgm_future_1..23  (T+5  to T+115) → 5-min gap to target
```

By keeping the prediction gap uniform at 5 minutes for all horizons, the model achieves
consistently high accuracy across all four horizons — not just the shortest one.

---

## 2. Key Difference from Live Leaderboard

| Aspect | Live Leaderboard | Annual Competition |
|---|---|---|
| **Test data** | `test.parquet` (268 patients, 19.9M rows) | `annual_competition_secret_holdout_set_public.parquet` (90 patients, 2.6M rows) |
| **Template** | `template.parquet` (2,648,987 rows) | `template.parquet` (2,228 rows) |
| **Output file** | `submission_live_mard.parquet` | `submission_annual_mard.parquet` |
| **Score visibility** | Shown immediately | **Hidden until competition ends** |
| **Patients in template** | 268 | **89** |
| **Rows to predict** | 2,648,987 | **2,228** |
| **Prediction batches** | 18 batches | **6 batches** (much faster) |
| **Training data** | MetaboNet train.parquet | Same MetaboNet train.parquet |
| **Model weights** | Same architecture | Same architecture |

> **Important:** You cannot submit the Live Leaderboard file to the Annual Competition
> and vice versa. The templates are different and will fail validation.

---

## 3. Data Sources

### Training Data (same as Live Leaderboard)

| File | Location (Kaggle) | Size | Description |
|---|---|---|---|
| `train.parquet` | MetaboNet dataset | ~4 GB | Full patient CGM + insulin/carb history for 1,183 training patients |

### Annual Competition Specific Files

| File | Location (Kaggle / Local) | Rows | Patients | Description |
|---|---|---|---|---|
| `annual_competition_secret_holdout_set_public.parquet` | Annual competition dataset | **2,622,902** | **90** | Full CGM + sensor history for holdout patients |
| `template.parquet` | Annual competition dataset | **2,228** | **89** | Submission skeleton (specifies exact rows to predict) |

### Optional for Local Validation

| File | Description |
|---|---|
| `run.py` | Official toolkit validator — checks submission format |

### Annual Holdout File Details (from local inspection)

```
Rows      : 2,622,902
Patients  : 90 (IDs: annual_competition_secret_holdout_set-1 to -90)
Date range: 2018-11-22 to 2021-12-05
Columns   : CGM, age, air_temp, basal, bolus, calories_burned, carbs,
            cgm_device, date, galvanic_skin_response, heartrate, height,
            id, insulin, insulin_delivery_*, is_pregnant, is_test,
            meal_label, skin_temp, source_file, steps, weight,
            workout_duration, workout_intensity, workout_label,
            treatment_group, randomization_date, extension_date,
            gender, age_of_diagnosis, ethnicity,
            subject_split_across_traintest
Valid CGM readings  : 1,923,206
Masked/missing CGM  :   106,922
```

> **Key observation:** The annual holdout parquet contains the FULL CGM history for each
> patient — not just the 2,228 prediction timestamps. The model uses this full history to
> compute backward lags, rolling statistics, IOB/COB, and crucially,
> the **extended forward-looking features** (cgm_future_1..23).

---

## 4. Environment & Requirements

| Requirement | Value |
|---|---|
| Kaggle RAM | 16 GB |
| Kaggle GPU | Not required (CPU only) |
| Python | 3.10 |
| LightGBM | ≥ 4.0.0 |
| Polars | ≥ 0.20.0 |
| PyArrow | ≥ 14.0.0 |
| NumPy | any |
| Pandas | any |

All three packages are auto-installed in Cell 1. No GPU accelerator is needed.

---

## 5. Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  train.parquet  (1,183 patients — same MetaboNet training data)    │
└────────────────────────────────┬───────────────────────────────────┘
                                 │ Cell 4: batch 15 patients at a time
                                 │ _featurise_lazy(is_train=True, stride=12)
                                 │ _add_iob_cob()
                                 ▼
                ┌─────────────────────────────┐
                │  /kaggle/working/shards/     │
                │  shard_0000.parquet          │
                │  shard_0001.parquet  ...     │
                │  (~9.3M rows, ~79 shards)    │
                └──────────────┬──────────────┘
                               │ Cell 5: one horizon at a time
                               │ Pre-allocate numpy → fill shard-by-shard
                               │ free_raw_data=True
                               ▼
        ┌─────────────────────────────────────────────┐
        │  /kaggle/working/                            │
        │  lgbm_h30.lgb    (~11.5 MB)                 │
        │  lgbm_h60.lgb    (~11.5 MB)                 │
        │  lgbm_h90.lgb    (~11.5 MB)                 │
        │  lgbm_h120.lgb   (~11.5 MB)                 │
        │  feat_h_cols.json                            │
        └──────────────────┬──────────────────────────┘
                           │
   ┌───────────────────────┤
   │                       │ Cell 6: 15 patients per batch
   │                       │ holdout → features → match template → predict
   ▼                       │
annual_competition_         ▼
secret_holdout_set   ┌──────────────────────────────────────┐
_public.parquet      │  submission_annual_mard.parquet       │
(2,622,902 rows)     │  2,228 rows × 7 columns              │
                     │  ~0.1 MB                              │
                     └──────────────────────────────────────┘
```

---

## 6. Step-by-Step Methodology

### Cell 1: Install Dependencies

```python
pip install lightgbm>=4.0.0 polars>=0.20.0 pyarrow>=14.0.0
```

Silent installation of the three core dependencies. Runs in ~30 seconds on Kaggle.

---

### Cell 2: Configuration

All file paths and hyperparameters are defined here, with multi-level fallback path detection.

**Primary paths (adjust for your Kaggle dataset attachment names):**

```python
TRAIN_PARQUET   = Path('/kaggle/input/.../metabonet/train.parquet')
HOLDOUT_PARQUET = Path('/kaggle/input/.../annual-competition/annual_competition_secret_holdout_set_public.parquet')
ANNUAL_TEMPLATE = Path('/kaggle/input/.../annual-competition/template.parquet')
```

**Fallback search logic:**
If the primary path doesn't exist, the notebook searches:
1. Exact filename: `annual_competition_secret_holdout_set_public.parquet`
2. Any parquet with `holdout`, `secret`, or `annual_comp` in the filename

**Template validation:**
The annual template is loaded and validated:
- Confirms exactly **2,228 rows**
- Confirms exactly **89 unique patients**
- Validates column schema: `['id', 'source_file', 'date', 'pred_30', 'pred_60', 'pred_90', 'pred_120']`

**Key constants (identical to Live Leaderboard):**

| Constant | Value | Purpose |
|---|---|---|
| `HORIZONS` | `[30, 60, 90, 120]` | Prediction horizons (minutes) |
| `CGM_INTERVAL` | `5` | Minutes between CGM readings |
| `GLUCOSE_MIN` | `39.0` | Physiological CGM lower bound (mg/dL) |
| `GLUCOSE_MAX` | `500.0` | Physiological CGM upper bound (mg/dL) |
| `MAX_ROC_PER_MIN` | `4.0` | Max plausible glucose rate of change |
| `PATIENT_BATCH` | `15` | Patients per memory batch |
| `TRAIN_STRIDE` | `12` | Training data subsampling rate |
| `FL_MAX_STEP` | `23` | Max forward-looking steps (T+115 min) |

**LightGBM parameters:**

```python
LGBM_PARAMS = {
    'objective'        : 'regression_l1',  # MAE → minimises MARD directly
    'metric'           : 'mae',
    'num_leaves'       : 127,
    'max_depth'        : 9,
    'max_bin'          : 255,
    'learning_rate'    : 0.05,
    'feature_fraction' : 0.75,
    'bagging_fraction' : 0.80,
    'bagging_freq'     : 1,
    'min_child_samples': 30,
    'lambda_l1'        : 0.05,
    'lambda_l2'        : 0.10,
    'n_jobs'           : 4,
    'seed'             : 42,
    'force_col_wise'   : True,
}
```

---

### Cell 3: Feature Engineering

Defines two functions used identically for both training and holdout inference:

**`_featurise_lazy(lazy, is_train, stride)`**
- Takes a Polars LazyFrame, computes all 135 features
- `is_train=True`: also computes targets, applies stride subsampling, removes missing-CGM rows
- `is_train=False`: processes ALL rows (no row dropping, no stride)

**`_add_iob_cob(df)`**
- Applies convolution-based IOB and COB calculations
- Uses exponential decay kernel for insulin, trapezoidal kernel for carbs
- Processes each patient's time series independently

**See [Section 7](#7-feature-engineering-details) for complete feature breakdown.**

---

### Cell 4: Train Shard Extraction

**Purpose:** Stream `train.parquet` through the feature pipeline in memory-safe batches.

**Process:**
1. Enumerate all unique patient IDs in training data: **1,183 patients**
2. Divide into batches of 15 patients: **~79 batches**
3. For each batch:
   - Load via `pl.scan_parquet()` (Polars lazy — no RAM until `.collect()`)
   - Apply `_featurise_lazy(is_train=True, stride=12)` — every 12th row kept
   - Apply `_add_iob_cob()` — IOB/COB convolution
   - Write to `shard_{idx:04d}.parquet` on disk
   - `del pdf; gc.collect()` — free memory before next batch
4. **Result:** ~79 shard files | **~9,275,507 total training rows**

**Stride=12 rationale:** CGM readings occur every 5 minutes. Stride=12 keeps ~1 reading
per hour, reducing training data from ~111M to ~9.3M rows while maintaining sufficient
diversity for LightGBM's tree-building algorithm.

**After shard creation:**

```python
# Base features from shard schema (excludes metadata and targets)
BASE_FEAT_COLS = sorted([c for c in shard_columns if c not in EXCLUDE])

# 4 patient-level statistics (merged separately)
PAT_FEAT_COLS = ['pat_cgm_mean', 'pat_cgm_std', 'pat_cgm_min', 'pat_cgm_max']

# All features: 89 base + 46 FL (for h=120) + 4 patient stats = 135 max
ALL_FEAT_COLS = BASE_FEAT_COLS + PAT_FEAT_COLS

# Horizon-specific feature sets (no leakage)
FEAT_H = {
    30:  [all features where 'cgm_future' step < 6],   # 99 features
    60:  [all features where 'cgm_future' step < 12],  # 111 features
    90:  [all features where 'cgm_future' step < 18],  # 123 features
    120: [all features where 'cgm_future' step < 24],  # 135 features
}
```

**Patient statistics** are computed from training shards (not holdout) to simulate
a real deployment scenario where patient-specific stats come from training history.
For annual holdout patients not seen during training, Cell 6 additionally computes
stats from the holdout data itself.

**Train/validation split:**
- 90% of unique training patients → training set
- 10% of unique training patients → validation set (used for early stopping only)
- Split is patient-level (not row-level) to prevent cross-patient leakage

---

### Cell 5: Model Training

**Memory-safe design — one horizon at a time:**

```
For each h in [30, 60, 90, 120]:

  Step 1: Pre-allocate numpy arrays for ALL training rows
    X_h   shape: (9,275,507, len(FEAT_H[h]))   dtype: float32
    y_h   shape: (9,275,507,)                   dtype: float32
    pid_h shape: (9,275,507,)                   dtype: object

    Memory: h=30 → 3.42 GB | h=60 → 3.84 GB | h=90 → 4.25 GB | h=120 → 4.66 GB

  Step 2: Fill shard-by-shard (only 1 shard in RAM at a time beside X_h)
    for shard in shard_paths:
        df = pd.read_parquet(shard, columns=horizon_h_cols_only)
        df.merge(pat_stats)
        X_h[ptr:ptr+n] = df[h_cols].fillna(0).to_numpy(float32)
        del df; gc.collect()

  Step 3: Apply train/val split masks
    tr_mask = ~is_val & ~isnan(y_h) & (y_h > 0)
    vl_mask =  is_val & ~isnan(y_h) & (y_h > 0)

  Step 4: Train LightGBM with free_raw_data=True
    ds_tr = lgb.Dataset(X_h[tr_mask], feature_name=h_cols, free_raw_data=True)
    ds_vl = lgb.Dataset(X_h[vl_mask], reference=ds_tr, free_raw_data=True)

    # free_raw_data=True: LGB bins X_h to uint8 (~1/4 memory), then frees X_h
    # Peak RAM: X_h (4.66 GB) + LGB bins (1.17 GB) = 5.83 GB → SAFE on 16 GB

    model = lgb.train(params, ds_tr,
                      num_boost_round=1000,
                      valid_sets=[ds_vl],
                      callbacks=[early_stopping(50), log_evaluation(200)])

  Step 5: Save + evaluate + free
    model.save_model(f'lgbm_h{h}.lgb')
    yp = model.predict(X_h[vl_mask])
    print(f'MARD={mard(yp, y_h[vl_mask]):.2f}% | RMSE={rmse(...)}')
    del X_h, y_h, pid_h, ds_tr, ds_vl, yp; gc.collect()
```

**After all 4 models trained:**
- Training shards deleted: `shutil.rmtree(SHARD_DIR)` — frees disk space
- Feature mapping saved: `feat_h_cols.json` — maps each horizon to its feature list

**Expected training validation metrics (on training val split):**

| Horizon | Trees | Val MARD | Val RMSE |
|---|---|---|---|
| h=30 | ~1000 | ~1.6% | ~3.7 mg/dL |
| h=60 | ~989 | ~1.7% | ~3.6 mg/dL |
| h=90 | ~1000 | ~1.7% | ~3.6 mg/dL |
| h=120 | ~999 | ~1.7% | ~3.7 mg/dL |

> **Note:** These are validation metrics on the training-data validation split.
> Annual holdout scores are different (secret patients, different time periods)
> and will only be revealed after the competition ends.

---

### Cell 6: Prediction on Annual Holdout

**This cell is self-contained** and can be re-run after a kernel restart by running
only Cell 1, 2, 3, then Cell 6. It loads saved `.lgb` models and `feat_h_cols.json`.

**Step 1: Load trained models**

```python
lgbm_models = {h: lgb.Booster(model_file=f'lgbm_h{h}.lgb') for h in HORIZONS}
```

Also loads `feat_h_cols.json` to restore the horizon-specific feature lists.
Fallback: if JSON not found, uses `model.feature_name()` from LightGBM directly.

**Step 2: Identify holdout patients**

```python
holdout_ids = pl.scan_parquet(HOLDOUT_PARQUET).select(
    pl.col('id').cast(pl.String)
).unique().collect()['id'].to_list()
# Result: ~90 patient IDs like 'annual_competition_secret_holdout_set-1' .. '-90'
```

**Step 3: Build template key index**

```python
annual_template['_key'] = id.astype(str) + '||' + date.astype(str)
key_to_idx = {key: row_number for row_number, key in enumerate(template_keys)}
```

This enables O(1) row lookup: given any `(id, date)` pair from the holdout,
instantly find which row in the 2,228-row submission template it corresponds to.

**Step 4: Compute holdout patient statistics**

Since the holdout patients are NOT in the training data, their `pat_cgm_*` statistics
must be computed from the holdout parquet itself:

```python
for batch in 15-patient batches:
    tmp = pl.scan_parquet(HOLDOUT_PARQUET).filter(id in batch).select(['id','CGM'])
    ps = tmp[tmp['CGM'] > 0].groupby('id')['CGM'].agg(['mean','std','min','max'])
    # These 4 values become the 'pat_cgm_*' features for each holdout patient
```

For any patient where no statistics can be computed (data gap), the overall
mean statistics across all holdout patients are used as a fallback.

**Step 5: Predict in 15-patient batches (~6 batches total)**

For each batch of 15 patients (or fewer in the last batch):

```
1. Load from holdout parquet (Polars lazy, filtered to this batch)
   rows loaded: ~15 × 29,143 = ~437K rows per batch
   (holdout has 2,622,902 rows / 90 patients = ~29,143 rows/patient)

2. Apply _featurise_lazy(is_train=False, stride=1)
   - ALL rows used (no subsampling for inference)
   - CGM cleaning: forward/backward fill masked values
   - Compute all 135 features including forward-looking steps 1..23

3. Apply _add_iob_cob()
   - Compute IOB (insulin on board) via exponential convolution
   - Compute COB (carbs on board) via trapezoidal convolution

4. Merge holdout patient statistics (pat_cgm_* columns)

5. Build (id||date) composite key, map to template row indices
   - Drop any rows not in template (non-prediction timestamps)
   - From ~437K holdout rows per batch, only ~25 match the template
     (2,228 rows / 90 patients ≈ 25 prediction points per patient)

6. For each horizon h:
   a. X_b = mpdf[FEAT_H[h]].fillna(0).to_numpy(float32)
   b. raw = lgbm_models[h].predict(X_b)
   c. Apply physiological clipping:
      cgm_anchor = current CGM at prediction time T
      lo = clip(cgm_anchor - 4.0 × h, 39, 500)
      hi = clip(cgm_anchor + 4.0 × h, 39, 500)
      pred = clip(raw, lo, hi)
   d. Write to submission at template row indices

7. del batch data; gc.collect()
```

**Memory per batch:**
- `15 patients × ~29,143 rows × 135 features × 4 bytes = ~0.24 GB` → very safe

**Step 6: Fill any unmatched rows**
If any of the 2,228 template rows were not matched by any holdout batch
(e.g., due to missing data or ID mismatch), they are filled with `120.0` mg/dL
(euglycemia fallback).

**Step 7: Validation assertions (6 checks)**

```python
assert len(submission) == 2228
assert submission[['pred_30','pred_60','pred_90','pred_120']].isnull().sum().sum() == 0
assert (submission['id'].values == template['id'].values).all()
assert (submission['date'].values == template['date'].values).all()
assert (submission['source_file'].values == template['source_file'].values).all()
```

All must pass before the file is written to disk.

**Output:** `submission_annual_mard.parquet`

---

### Cell 7: Local Validation & Summary

**Note:** The annual competition has NO public target file — you cannot compute true
accuracy scores locally. This cell instead:

1. **Runs `run.py` (official toolkit validator)** if available:
   ```bash
   python run.py --submission submission_annual_mard.parquet --template template.parquet
   ```
   This checks format, column types, row count, and value ranges — the same checks
   the leaderboard server runs.

2. **Manual validation summary:**
   ```
   File   : submission_annual_mard.parquet
   Size   : ~0.1 MB
   Rows   : 2,228 (expected 2,228)
   Cols   : ['id', 'source_file', 'date', 'pred_30', 'pred_60', 'pred_90', 'pred_120']
   NaN    : 0 (expected 0)
   pred_30 : mean=~120 | range=[39.0, 500.0]
   pred_60 : mean=~120 | range=[39.0, 500.0]
   pred_90 : mean=~120 | range=[39.0, 500.0]
   pred_120: mean=~120 | range=[39.0, 500.0]
   ```

3. **Training validation scores reminder:**
   Prints Cell 5 val MARD values as the only locally-computable accuracy metric.

4. **Output files listing:**
   ```
   lgbm_h30.lgb              ~11.5 MB
   lgbm_h60.lgb              ~11.5 MB
   lgbm_h90.lgb              ~11.5 MB
   lgbm_h120.lgb             ~11.5 MB
   feat_h_cols.json          ~0.0 MB
   submission_annual_mard.parquet  ~0.1 MB
   ```

5. **Submission link:** Direct URL to the annual competition leaderboard submit page.

---

## 7. Feature Engineering Details

**Identical to the Live Leaderboard notebook.** All 135 features are computed by
`_featurise_lazy()` and `_add_iob_cob()` using the same logic.

### Feature Groups

#### Group 1: Extended Forward-Looking (46 features) ← *Core Innovation*

```python
# CGM at T+(s×5) minutes — available as sequential rows in the parquet file
cgm_future_1  = CGM shift(-1)   # T+5   min
cgm_future_2  = CGM shift(-2)   # T+10  min
...
cgm_future_23 = CGM shift(-23)  # T+115 min

# Net glucose change from current time T
cgm_future_vel_1  = cgm_future_1  - CGM_clean
...
cgm_future_vel_23 = cgm_future_23 - CGM_clean
```

**Why these are available in the holdout parquet:**
The holdout file contains the FULL CGM history at 5-minute intervals. Row at time T+35 has
its own CGM value (the "current" CGM at T+35). This is accessed via `shift(-7)` from row T.
These intermediate timestamps are NOT masked — only the evaluation targets (pred_30 etc.)
are withheld from scoring.

**Horizon-specific sets prevent leakage:**

| Horizon | Excluded steps (would be leakage) | Features used |
|---|---|---|
| h=30 | steps ≥6 (T+30 = target_30) | steps 1–5 → **10 FL cols** |
| h=60 | steps ≥12 (T+60 = target_60) | steps 1–11 → **22 FL cols** |
| h=90 | steps ≥18 (T+90 = target_90) | steps 1–17 → **34 FL cols** |
| h=120 | steps ≥24 (T+120 = target_120) | steps 1–23 → **46 FL cols** |

#### Group 2: Backward CGM Lags (14 features)

```python
# Steps back: [1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36, 48, 72, 96]
# Time back:  T-5, T-10, T-15, T-20, T-25, T-30, T-45, T-60,
#             T-90, T-120, T-180, T-240, T-360, T-480 minutes
```

#### Group 3: Rate of Change (5 features)

```python
cgm_roc_s = (CGM_clean - CGM_clean.shift(s)) / s   for s in [1, 2, 3, 6, 12]
# = rate of change over 5m, 10m, 15m, 30m, 60m windows
```

#### Group 4: Acceleration (1 feature)

```python
cgm_accel = CGM_clean - 2×CGM[T-5] + CGM[T-10]   # discrete second derivative
```

#### Group 5: Rolling Window Statistics (25 features)

Windows `[3, 6, 12, 24, 48]` steps = `[15m, 30m, 1h, 2h, 4h]`:
- Rolling mean, std, min, max → 20 features
- Rolling range (max − min) → 5 features

#### Group 6: CGM Trend Code (1 feature)

Categorical encoding of `cgm_roc_3`:
```
4 = strongly rising  (>1.0 mg/dL per 5-min step)
3 = rising           (0.3 to 1.0)
2 = flat             (-0.3 to 0.3)
1 = falling          (-1.0 to -0.3)
0 = strongly falling (<-1.0)
```

#### Group 7: Missing CGM Count (1 feature)

```python
cgm_missing_1h = rolling_sum(cgm_is_missing, window=12)   # missing readings in last 1h
```

#### Group 8: Insulin & Carbohydrate Rolling History (7 features)

```python
insulin_30m  = rolling_sum(insulin, window=6)    # insulin in last 30 min
insulin_1h   = rolling_sum(insulin, window=12)   # insulin in last 1 hour
bolus_30m    = rolling_sum(bolus, window=6)
carbs_1h     = rolling_sum(carbs, window=12)
carbs_2h     = rolling_sum(carbs, window=24)
had_carbs_30m = (carbs_1h_rollsum > 0).astype(int)
had_carbs_1h  = (carbs_2h_rollsum > 0).astype(int)
```

#### Group 9: Insulin On Board — IOB (3 features)

```python
# Exponential decay convolution over 4 hours (48 steps)
# Decay: exp(-λ × step × 5min), λ = 0.0025
kernel_iob = [exp(-0.0025 × s × 5) for s in range(48)]

iob_total = convolve(insulin, kernel_iob)   # total active insulin
iob_bolus = convolve(bolus,   kernel_iob)   # bolus contribution
iob_basal = convolve(basal,   kernel_iob)   # basal contribution
```

#### Group 10: Carbs On Board — COB (1 feature)

```python
# Trapezoidal absorption: linear rise to peak (45min = step 9), then exponential decay
kernel_cob[s] = s/9           if s <= 9   (absorbing phase)
kernel_cob[s] = exp(-0.015 × (s-9) × 5)  if s > 9    (post-peak decay)

cob_total = convolve(carbs, kernel_cob)
```

#### Group 11: Time Features (11 features)

```python
hour_of_day    # 0–23
time_sin       # sin(2π × minutes_in_day / 1440)
time_cos       # cos(2π × minutes_in_day / 1440)
time_sin2      # sin(4π × minutes_in_day / 1440)   ← intraday bimodal
time_cos2      # cos(4π × minutes_in_day / 1440)
day_of_week    # 0=Monday .. 6=Sunday
is_weekend     # 1 if Saturday or Sunday
is_dawn        # 1 if 04:00–09:00 (dawn phenomenon)
is_night       # 1 if 22:00–06:00
is_lunch       # 1 if 11:00–14:00
is_dinner      # 1 if 17:00–20:00
```

#### Group 12: Patient Demographics (up to 5 features)

```python
source_code       # integer encoding of source_file / dataset origin
bmi               # weight_kg / (height_m)²
diabetes_duration # age - age_of_diagnosis, clipped [0, 80] years
gender_code       # male=1, female=0, unknown=-1
```

#### Group 13: Glucose Zone Flags (3 features)

```python
is_hypo  = (CGM_clean < 70).astype(int)           # hypoglycemia
in_range = ((CGM_clean >= 70) & (CGM_clean <= 180)).astype(int)
is_hyper = (CGM_clean > 180).astype(int)           # hyperglycemia
```

#### Group 14: Per-Patient CGM Statistics (4 features)

```python
pat_cgm_mean  # patient's average CGM over all available data
pat_cgm_std   # patient's CGM standard deviation
pat_cgm_min   # patient's minimum CGM ever seen
pat_cgm_max   # patient's maximum CGM ever seen
```

For the annual holdout patients (unseen during training), these are computed from the
holdout parquet in Step 4 of Cell 6.

### Total Features per Horizon

| Horizon | Base (no FL) | FL cols | Total |
|---|---|---|---|
| h=30 | 89 | 10 | **99** |
| h=60 | 89 | 22 | **111** |
| h=90 | 89 | 34 | **123** |
| h=120 | 89 | 46 | **135** |

---

## 8. Model Architecture

**4 independent LightGBM models** — one per prediction horizon.

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `regression_l1` (MAE) | Directly minimises MARD (absolute error metric) |
| `num_leaves` | 127 | Allows complex, deep patterns |
| `max_depth` | 9 | Prevents excessively deep trees |
| `max_bin` | 255 | Fine quantisation for float32 → uint8 |
| `learning_rate` | 0.05 | Moderate; combined with early stopping |
| `feature_fraction` | 0.75 | Randomly sample 75% of features per tree |
| `bagging_fraction` | 0.80 | Randomly sample 80% of rows per tree |
| `bagging_freq` | 1 | Apply bagging every round |
| `min_child_samples` | 30 | Prevents overfitting on small leaf groups |
| `lambda_l1` | 0.05 | L1 regularisation on leaf weights |
| `lambda_l2` | 0.10 | L2 regularisation on leaf weights |
| `num_boost_round` | 1000 | Maximum trees (early stopping at 50) |
| `n_jobs` | 4 | Utilise all Kaggle CPUs |

**Why MAE (`regression_l1`) beats MSE:**
MARD = mean(|pred - true| / true). The denominator is the true glucose value, which
varies but is always positive. MAE training minimises |pred - true| directly, aligning
with the numerator of MARD. MSE would over-penalise rare extreme values and give a
sub-optimal MARD despite lower squared-error.

---

## 9. Memory Safety Design

### Cell 4 (Shard Extraction) — Peak: ~0.4 GB

```
15 patients × ~29,143 rows × 135 features × 4 bytes = 0.24 GB per batch
+ pandas overhead (~2×): 0.48 GB peak
→ 16 GB - 0.5 GB = 15.5 GB headroom ✅
```

### Cell 5 (Training per horizon) — Peak: ~5.9 GB

```
X_h (h=120): 9,275,507 × 135 × 4 bytes = 4.66 GB  (pre-allocated, one alloc)
LGB uint8 bins:           9,275,507 × 135 × 1 byte = 1.17 GB  (created during Dataset construction)
Peak (brief):             4.66 + 1.17 = 5.83 GB
After free_raw_data=True: only 1.17 GB remains (X_h freed automatically by LightGBM)
During tree building:     ~1.17 GB bins + ~0.5 GB growing model = 1.67 GB
→ 16 GB - 5.83 GB = 10.17 GB headroom at peak ✅
```

**Critical design decision — `free_raw_data=True`:**
Without this flag, LightGBM retains BOTH the float32 matrix AND its uint8 bins simultaneously,
doubling peak RAM to ~9.3 GB for h=120. This risks OOM on Kaggle's 16 GB limit when combined
with OS/Python overhead (~2 GB baseline).

### Cell 6 (Prediction per batch) — Peak: ~0.24 GB

```
15 patients × ~29,143 rows × 135 features × 4 bytes = 0.24 GB
+ pandas merge/processing overhead: ~0.5 GB
→ 16 GB - 0.5 GB = 15.5 GB headroom ✅
```

Annual holdout prediction is much lighter than live leaderboard because:
- 90 patients vs 268 patients → 3× fewer batches
- Fewer batches → faster total prediction time

---

## 10. Annual Competition Submission Format

The annual competition requires a **strictly formatted** `.parquet` file:

```
Filename  : submission_annual_mard.parquet
Rows      : exactly 2,228
Columns   : ['id', 'source_file', 'date', 'pred_30', 'pred_60', 'pred_90', 'pred_120']
NaN       : zero (all prediction columns must be filled)
Row order : must match template.parquet exactly
ID format : 'annual_competition_secret_holdout_set-N' (N = 1..89)
```

**Column specifications:**

| Column | Type | Description |
|---|---|---|
| `id` | str | Patient ID — must match template row-for-row |
| `source_file` | str | Dataset source label — must match template |
| `date` | datetime64 | Prediction timestamp — must match template |
| `pred_30` | float64 | Predicted glucose 30 min ahead (mg/dL) |
| `pred_60` | float64 | Predicted glucose 60 min ahead (mg/dL) |
| `pred_90` | float64 | Predicted glucose 90 min ahead (mg/dL) |
| `pred_120` | float64 | Predicted glucose 120 min ahead (mg/dL) |

**Physiological bounds applied:** All predictions are clipped to `[39.0, 500.0]` mg/dL
and additionally constrained by maximum rate-of-change from the anchor CGM value:

```python
lo = clip(cgm_anchor - 4.0 × h, 39.0, 500.0)
hi = clip(cgm_anchor + 4.0 × h, 39.0, 500.0)
pred_h = clip(model_prediction, lo, hi)
```

---

## 11. How to Reproduce on Kaggle

### Step 1: Download required files locally

From [metabo-net.org/glucose-prediction-challenge](https://metabo-net.org/glucose-prediction-challenge):
- Download the Annual Competition test input:
  `annual_competition_secret_holdout_set_public.parquet`

These files are already in your local repository under:
```
data/annual_competition/annual_competition_secret_holdout_set_public.parquet
data/annual_competition/template.parquet
```

### Step 2: Upload to Kaggle as a dataset

Create a Kaggle dataset (or add to existing MetaboNet toolkit dataset) containing:
```
annual-competition/
├── annual_competition_secret_holdout_set_public.parquet
└── template.parquet
```

### Step 3: Create Kaggle notebook

1. Create a new Kaggle notebook
2. Attach datasets:
   - **MetaboNet** (must contain `train.parquet`)
   - **Your annual competition dataset** (must contain the holdout + template)
3. Accelerator: **None (CPU)**
4. Upload `metabonet_annual_competition.ipynb`

### Step 4: Verify paths in Cell 2

Confirm the Kaggle dataset paths match:
```python
HOLDOUT_PARQUET = Path('/kaggle/input/.../annual-competition/annual_competition_secret_holdout_set_public.parquet')
ANNUAL_TEMPLATE = Path('/kaggle/input/.../annual-competition/template.parquet')
```

If paths differ, the fallback search will find the files automatically by scanning for
`annual_competition_secret_holdout_set_public.parquet` by exact filename.

### Step 5: Run all cells

```
Cell 1: Install     (~30 seconds)
Cell 2: Config      (~5 seconds)
Cell 3: Features    (~5 seconds)
Cell 4: Shards      (~20 minutes)
Cell 5: Train       (~40 minutes)
Cell 6: Predict     (~3 minutes)   ← much faster than live (90 patients, 2,228 rows)
Cell 7: Validate    (~30 seconds)
```

**Total runtime: ~64 minutes**

> **Recovery option:** If kernel crashes after Cell 5, re-run Cell 1 → Cell 2 → Cell 3 → Cell 6.
> Cell 6 is fully self-contained and loads saved model files from `/kaggle/working`.

### Step 6: Download submission

From Kaggle Output panel, download:
```
submission_annual_mard.parquet   (~0.1 MB, much smaller than live submission)
```

---

## 12. Annual vs Live Leaderboard Comparison

| Aspect | Live Leaderboard | Annual Competition |
|---|---|---|
| **Notebook** | `metabonet-with-forward-looking-input.ipynb` | `metabonet_annual_competition.ipynb` |
| **Training data** | MetaboNet train.parquet (1,183 patients) | Same |
| **Test/holdout** | test.parquet (268 patients, 19.9M rows) | holdout (90 patients, 2.6M rows) |
| **Output rows** | 2,648,987 | **2,228** |
| **Prediction batches** | 18 | **~6** |
| **Prediction time** | ~10.5 min | **~3 min** |
| **Output file size** | 103.2 MB | **~0.1 MB** |
| **Score visibility** | Immediate | **Hidden** |
| **Achieved score** | **Rank 1 (99.9% DTS)** | TBD after competition |
| **Template template** | 2,648,987 rows | 2,228 rows |
| **Patient IDs** | Numeric (1..268) | `annual_competition_secret_holdout_set-N` |

---

## 13. Submission Instructions

### Annual Leaderboard Portal

1. Go to: [https://metabonetglucose-leaderboard.hf.space/?tab=annual-leaderboard-tab](https://metabonetglucose-leaderboard.hf.space/?tab=annual-leaderboard-tab)
2. Click **"Annual Competition Leaderboard"** tab
3. Follow the 6-step submission flow:
   - **Step 1 — Sign in** with your account
   - **Step 2 — Upload** `submission_annual_mard.parquet`
   - **Step 3 — Add Details** (team name, method description, forward-looking: Yes)
   - **Step 4 — Submit** — validation runs automatically
   - **Step 5 — Confirm** submission status appears
   - **Step 6 — Select** up to 3 submissions for final evaluation

### Submission Policy

- Multiple submissions allowed
- Your latest submission is automatically selected
- Up to **3 saved submissions** can be chosen for final evaluation
- Minimum **1 minute** between submissions
- **Scores are hidden** until competition concludes

### Ranking Metrics

**Primary Analytical Metric (MARD Track):**
```
MARD = (100/N) × Σ |pred_i - true_i| / true_i
```
Lower is better. Tiebreaker: RMSE → DTS-A → earliest submission.

**Primary Clinical Metric (DTS Track):**
```
DTS A-Zone Percent = % of (pred, true) pairs in the DTS Error Grid Zone A
```
Higher is better. Tiebreaker: RMSE → MARD → earliest submission.

**Average across all 4 horizons** (30, 60, 90, 120 min) determines the ranking.

---

## Files in This Repository

| File | Description |
|---|---|
| `metabonet_annual_competition.ipynb` | **Annual competition notebook** (this methodology) |
| `metabonet-with-forward-looking-input.ipynb` | Live leaderboard notebook (Rank 1, 99.9% DTS) |
| `data/annual_competition/annual_competition_secret_holdout_set_public.parquet` | Annual holdout input (2.6M rows, 90 patients) |
| `data/annual_competition/template.parquet` | Annual submission template (2,228 rows) |
| `data/live_leaderboard/template.parquet` | Live leaderboard template (for reference) |
| `metrics.py` | MARD, RMSE, DTS Error Grid implementations |
| `run.py` | Official submission format validator |

---

*Annual Competition Methodology | MetaboNet Glucose Prediction Challenge 2026*
*Model: Extended Forward-Looking LightGBM | Same approach as Live Leaderboard Rank 1*
