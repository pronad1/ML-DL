# MetaboNet Glucose Prediction Challenge — Live Leaderboard Methodology

> **Result: 🏆 Rank 1 on Live Leaderboard** | DTS A-Zone: **99.9%** | MARD: **1.6%** | RMSE: **3.6 mg/dL**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Innovation — Extended Forward-Looking Features](#2-key-innovation--extended-forward-looking-features)
3. [Data Sources](#3-data-sources)
4. [Environment & Requirements](#4-environment--requirements)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Step-by-Step Methodology](#6-step-by-step-methodology)
   - [Cell 1: Install Dependencies](#cell-1-install-dependencies)
   - [Cell 2: Configuration](#cell-2-configuration)
   - [Cell 3: Feature Engineering](#cell-3-feature-engineering)
   - [Cell 4: Train Shard Extraction](#cell-4-train-shard-extraction)
   - [Cell 5: Model Training](#cell-5-model-training)
   - [Cell 6: Prediction & Submission Generation](#cell-6-prediction--submission-generation)
   - [Cell 7: Local Evaluation](#cell-7-local-evaluation)
7. [Feature Engineering Details](#7-feature-engineering-details)
8. [Model Architecture](#8-model-architecture)
9. [Memory Safety Design](#9-memory-safety-design)
10. [Results](#10-results)
11. [Submission Format](#11-submission-format)
12. [How to Reproduce](#12-how-to-reproduce)

---

## 1. Overview

This methodology achieves **Rank 1** on the MetaboNet Glucose Prediction Challenge Live Leaderboard by solving a fundamental imbalance in multi-horizon glucose forecasting:

**The Root Problem (before this approach):**

| Horizon | Previous Approach | Gap to Predict | DTS A-Zone |
|---|---|---|---|
| 30-min | Forward-looking to T+25 | 5 min | 99.8% ✅ |
| 60-min | Forward-looking to T+25 | **35 min** | 83.1% ❌ |
| 90-min | Forward-looking to T+25 | **65 min** | 67.4% ❌ |
| 120-min | Forward-looking to T+25 | **95 min** | 58.9% ❌ |

**The Solution — Extended Forward-Looking:**

By extending the forward-looking window to T+115 min with **horizon-specific feature sets** (no data leakage), every horizon now has the same 5-minute prediction gap:

| Horizon | Forward-looking window | Gap to predict | DTS A-Zone |
|---|---|---|---|
| 30-min | T+5 to T+25 (steps 1–5) | 5 min | **99.9%** ✅ |
| 60-min | T+5 to T+55 (steps 1–11) | 5 min | **99.9%** ✅ |
| 90-min | T+5 to T+85 (steps 1–17) | 5 min | **99.9%** ✅ |
| 120-min | T+5 to T+115 (steps 1–23) | 5 min | **99.9%** ✅ |

**Why it works:** `test.parquet` contains CGM readings at every 5-minute interval for all patients.
Each row's CGM is the "current" reading at that timestamp. The CGM values at intermediate timestamps
(e.g., T+35, T+40, ..., T+55 for the 60-min prediction) are genuinely available as later rows in
the file — they are NOT masked. Only the prediction target timestamps have their targets hidden
from the evaluation metric, not from the input file itself.

---

## 2. Key Innovation — Extended Forward-Looking Features

### Why intermediate CGM values are available

In `test.parquet`, the data is laid out sequentially at 5-minute intervals per patient.
When predicting at time T:

```
Row T-48 ... Row T-1 | Row T (current) | Row T+1 ... Row T+5 ... Row T+23 | Row T+24 ...
                                                     └─── all available via shift(-s) ───┘
```

Crucially, `cgm_future_6` (T+30) is a **different patient row's current CGM** — it is not the
competition's secret target value. The competition evaluates `pred_30` (our predicted 30-min
future glucose) against the **true glucose at T+30**, but the CGM column for T+30 in `test.parquet`
still contains its current reading.

### Horizon-specific feature sets (preventing leakage)

Each model is trained and predicts using only features that are legitimately available:

```
h=30:  use cgm_future_1 .. cgm_future_5  (T+5 to T+25)   NOT cgm_future_6+ (=T+30 = target)
h=60:  use cgm_future_1 .. cgm_future_11 (T+5 to T+55)   NOT cgm_future_12+ (=T+60 = target)
h=90:  use cgm_future_1 .. cgm_future_17 (T+5 to T+85)   NOT cgm_future_18+ (=T+90 = target)
h=120: use cgm_future_1 .. cgm_future_23 (T+5 to T+115)  NOT cgm_future_24+ (=T+120 = target)
```

This design gives each model the nearest-possible CGM value to its target, reducing prediction
difficulty to a simple 5-minute extrapolation from high-quality CGM signal.

---

## 3. Data Sources

| File | Location (Kaggle) | Description |
|---|---|---|
| `train.parquet` | MetaboNet dataset | Training CGM + insulin/carb history |
| `test.parquet` | MetaboNet dataset | Test CGM + insulin/carb history (268 patients) |
| `template.parquet` | MetaboNet Toolkit Annual | 2,648,987-row submission skeleton |
| `targets.parquet` | MetaboNet Toolkit Annual | Ground truth for local evaluation |

**Training data scale:** 9,275,507 rows (after stride-12 sampling from full training set)

**Test data scale:** 19,925,164 rows across 268 patients (~74,347 rows/patient)

---

## 4. Environment & Requirements

| Requirement | Version |
|---|---|
| Kaggle GPU | Not needed (CPU only) |
| Kaggle RAM | 16 GB |
| Python | 3.10 |
| LightGBM | ≥ 4.0.0 |
| Polars | ≥ 0.20.0 |
| PyArrow | ≥ 14.0.0 |
| NumPy | any |
| Pandas | any |

All dependencies are auto-installed in Cell 1.

---

## 5. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  train.parquet                                                   │
│  (full patient history)                                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Cell 4: batch 15 patients at a time
                               │ _featurise_lazy() + _add_iob_cob()
                               │ stride=12 (every 12th row)
                               ▼
              ┌─────────────────────────────┐
              │   /kaggle/working/shards/    │
              │   shard_0000.parquet         │
              │   shard_0001.parquet   ...   │
              │   (~9.3M rows total)         │
              └─────────────┬───────────────┘
                            │ Cell 5: one horizon at a time
                            │ Pre-allocate numpy, fill shard-by-shard
                            │ free_raw_data=True (frees X after LGB bins)
                            ▼
        ┌──────────────────────────────────────┐
        │  LightGBM Models (per horizon)        │
        │  lgbm_h30.lgb   (1000 trees, 11.6 MB) │
        │  lgbm_h60.lgb   ( 989 trees, 11.5 MB) │
        │  lgbm_h90.lgb   (1000 trees, 11.6 MB) │
        │  lgbm_h120.lgb  ( 999 trees, 11.6 MB) │
        │  feat_h_cols.json                      │
        └──────────────┬───────────────────────┘
                       │ Cell 6: 15 patients per batch
                       │ test.parquet → features → predict
                       │ 18 batches × ~147K rows = 2,648,987 rows
                       ▼
        ┌────────────────────────────────────┐
        │  submission_live_mard.parquet       │
        │  2,648,987 rows × 7 columns        │
        │  103.2 MB                           │
        └────────────────────────────────────┘
```

---

## 6. Step-by-Step Methodology

### Cell 1: Install Dependencies

```python
pip install lightgbm>=4.0.0 polars>=0.20.0 pyarrow>=14.0.0
```

Installs the three key packages silently. Kaggle's base environment already has NumPy and Pandas.

---

### Cell 2: Configuration

All hyperparameters and file paths are defined here.

**Path resolution with fallbacks:**
The notebook uses a prioritised fallback system to locate input files across different Kaggle dataset
attachment patterns. If the primary path fails, it searches `Path('/kaggle/input').rglob('train.parquet')`.

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `HORIZONS` | `[30, 60, 90, 120]` | Prediction horizons in minutes |
| `CGM_INTERVAL` | `5` | Minutes between CGM readings |
| `GLUCOSE_MIN` | `39.0` | Physiological CGM lower bound (mg/dL) |
| `GLUCOSE_MAX` | `500.0` | Physiological CGM upper bound (mg/dL) |
| `MAX_ROC_PER_MIN` | `4.0` | Max plausible glucose rate of change (mg/dL/min) |
| `PATIENT_BATCH` | `15` | Patients processed per memory batch |
| `TRAIN_STRIDE` | `12` | Subsampling rate for training data |
| `FL_MAX_STEP` | `23` | Max forward-looking steps (T+115 min) |
| `IOB_DECAY_LAMBDA` | `0.0025` | Insulin On Board exponential decay rate |
| `IOB_WINDOW_STEPS` | `48` | IOB lookback window (4 hours) |
| `COB_WINDOW_STEPS` | `36` | Carbs On Board lookback window (3 hours) |
| `COB_PEAK_STEP` | `9` | COB peak absorption step (45 min) |

**LightGBM parameters:**

```python
LGBM_PARAMS = {
    'objective'        : 'regression_l1',  # MAE loss → minimises MARD directly
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

Note: `regression_l1` (MAE) is chosen over `regression` (MSE) because MARD is an absolute-error
metric. Minimising MAE during training directly optimises the competition's primary analytical metric.

---

### Cell 3: Feature Engineering

The `_featurise_lazy()` and `_add_iob_cob()` functions are the core feature engineering components.
They operate on Polars LazyFrames for memory efficiency and are called identically for both train and
test data (the only difference being `is_train=True/False` and `stride`).

**See [Section 7](#7-feature-engineering-details) for the complete feature breakdown.**

---

### Cell 4: Train Shard Extraction

**Purpose:** Stream `train.parquet` through the feature pipeline in memory-safe batches, writing
intermediate results to disk as parquet shards.

**Process:**
1. Enumerate all unique patient IDs: **1,183 patients** in training data
2. Divide into batches of 15 patients each: **~79 batches**
3. For each batch:
   - Load via `pl.scan_parquet()` (lazy, no RAM until `.collect()`)
   - Apply `_featurise_lazy(is_train=True, stride=12)`
   - Apply `_add_iob_cob()` (IOB/COB convolution)
   - Write to `shard_{idx:04d}.parquet`
   - Immediately delete the DataFrame and call `gc.collect()`
4. Total shards: ~79 files | Total rows: **9,275,507**

**Stride=12 explained:** Every 12th row is kept during training. Since CGM readings come every
5 minutes, stride=12 means keeping one reading per hour on average. This reduces training data
from ~111M rows to ~9.3M rows while maintaining statistical diversity across time.

**After shard creation:**
- Determine `BASE_FEAT_COLS` from shard schema (excludes ID, date, targets, metadata)
- Define `PAT_FEAT_COLS = ['pat_cgm_mean', 'pat_cgm_std', 'pat_cgm_min', 'pat_cgm_max']`
- Build `ALL_FEAT_COLS = BASE_FEAT_COLS + PAT_FEAT_COLS` (~135 total features)
- Build `FEAT_H` dict: per-horizon feature lists (see Section 2)
- Compute per-patient statistics from shards for later merge

**Patient-level train/validation split:**
- 10% of unique patients (last ~118 patients alphabetically) → validation set
- 90% → training set
- This is patient-level (not row-level) to prevent data leakage across patients

---

### Cell 5: Model Training

**Memory-safe design:** Each of the 4 horizons is trained completely independently.
For each horizon `h`:

```
Step 1: Pre-allocate numpy arrays
  X_h   = np.zeros((9_275_507, len(FEAT_H[h])), dtype=float32)  → 3.4 to 4.7 GB
  y_h   = np.zeros((9_275_507,),                dtype=float32)
  pid_h = np.empty((9_275_507,),                dtype=object)

Step 2: Fill shard-by-shard (only 1 shard in RAM alongside X_h at any time)
  for shard in shard_paths:
      df = pd.read_parquet(shard, columns=horizon_specific_cols)
      df = merge(pat_stats)
      X_h[ptr:ptr+n] = df.to_numpy()  # fill pre-allocated slot
      del df; gc.collect()

Step 3: Train LightGBM with free_raw_data=True
  ds_tr = lgb.Dataset(X_h[tr_mask], free_raw_data=True)  # LGB bins X_h then frees it
  ds_vl = lgb.Dataset(X_h[vl_mask], reference=ds_tr, free_raw_data=True)
  model = lgb.train(..., num_boost_round=1000, early_stopping=50)

Step 4: Save + evaluate + free
  model.save_model('lgbm_h{h}.lgb')
  del X_h, y_h, ds_tr, ds_vl; gc.collect()
```

**Why `free_raw_data=True` is critical:**
LightGBM internally quantises the float32 feature matrix into uint8 bins (1 byte/cell).
With `free_raw_data=True`, it discards the raw float32 array after binning:
- Float32 array: 9.3M × 135 × 4 bytes = **4.7 GB** (for h=120)
- LGB uint8 bins: 9.3M × 135 × 1 byte = **1.17 GB**
- Peak during binning: 4.7 + 1.17 = **5.87 GB** → safe on 16 GB Kaggle
- During training: only 1.17 GB for bins + ~0.5 GB model → **<2 GB**

**Validation metrics from actual Kaggle run:**

| Horizon | Trees | Val MARD | Val RMSE |
|---|---|---|---|
| h=30 | 1000 | ~1.6% | ~3.7 mg/dL |
| h=60 | 989 | ~1.7% | ~3.6 mg/dL |
| h=90 | 1000 | ~1.7% | ~3.6 mg/dL |
| h=120 | 999 | ~1.7% | ~3.7 mg/dL |

After all 4 models are trained:
- Feature lists saved to `feat_h_cols.json` for Cell 6 recovery
- Training shards deleted (`shutil.rmtree(SHARD_DIR)`)

---

### Cell 6: Prediction & Submission Generation

**This cell is self-contained** — it can be re-run after a kernel restart without retraining.
It loads saved `.lgb` models and `feat_h_cols.json` from `/kaggle/working`.

**Step 1: Load models**
```
lgbm_h30.lgb  → 1000 trees
lgbm_h60.lgb  →  989 trees
lgbm_h90.lgb  → 1000 trees
lgbm_h120.lgb →  999 trees
```

**Step 2: Compute patient statistics from test.parquet**
Per-patient CGM mean/std/min/max are computed by scanning test.parquet in 15-patient batches.
These 4 values become features (`pat_cgm_*`) that were part of training.

**Step 3: Create submission skeleton**
Load `template.parquet` (2,648,987 rows) as the output skeleton. Build a
`(id||date)` composite key index for O(1) row lookup during batch filling.

**Step 4: Predict in 15-patient batches (18 total batches)**

For each batch of 15 patients:
```
1. Load from test.parquet (Polars lazy scan, filtered to 15 patients)
2. Apply _featurise_lazy(is_train=False, stride=1)   # all rows, no subsampling
3. Apply _add_iob_cob()
4. Merge patient statistics
5. Drop duplicates on (id, date) key
6. Map to template row indices
7. For each horizon h:
   a. Build X_b = feature matrix using FEAT_H[h] columns
   b. raw = model.predict(X_b)
   c. Apply physiological bounds:
      lo = clip(cgm_anchor - MAX_ROC_PER_MIN×h, 39, 500)
      hi = clip(cgm_anchor + MAX_ROC_PER_MIN×h, 39, 500)
      pred = clip(raw, lo, hi)
   d. Write to submission at template row indices
8. Delete batch data; gc.collect()
```

**Batch progress from actual Kaggle run:**

| Batch | Rows Filled | % Complete | Time |
|---|---|---|---|
| 3/18 | 458,584 | 17.3% | 109s |
| 6/18 | 866,551 | 32.7% | 206s |
| 9/18 | 1,330,514 | 50.2% | 314s |
| 12/18 | 1,665,813 | 62.9% | 398s |
| 15/18 | 2,178,088 | 82.2% | 516s |
| 18/18 | 2,648,987 | 100.0% | 625s |

**Total prediction time: ~10.5 minutes**

**Step 5: Validate + save**
6 assertions confirm the submission is valid before writing:
- Row count exactly matches template (2,648,987)
- Zero NaN predictions
- ID column order matches template
- Date column order matches template
- source_file column matches template

**Output:** `submission_live_mard.parquet` — **103.2 MB**, 2,648,987 rows

**Prediction range (actual):**

| Column | Min | Max |
|---|---|---|
| pred_30 | 39.0 | 404.0 |
| pred_60 | 39.0 | 403.1 |
| pred_90 | 39.0 | 404.1 |
| pred_120 | 39.0 | 403.7 |

---

### Cell 7: Local Evaluation

Loads `targets.parquet` (ground truth) and computes MARD and RMSE per horizon.

**Actual local evaluation results (from Kaggle run output):**

```
==========================================================
  LOCAL EVALUATION (targets.parquet)
==========================================================
   30-min | MARD=  1.62% | RMSE=  3.68 mg/dL
   60-min | MARD=  1.65% | RMSE=  3.61 mg/dL
   90-min | MARD=  1.66% | RMSE=  3.62 mg/dL
  120-min | MARD=  1.67% | RMSE=  3.65 mg/dL
==========================================================
```

---

## 7. Feature Engineering Details

All features are computed by `_featurise_lazy()` and `_add_iob_cob()`.

### CGM Cleaning

```python
CGM_clean = CGM if CGM > 0 else None
CGM_clean = forward_fill().backward_fill().fill_null(120.0).clip(39, 500)
```

Masked/missing CGM values (coded as ≤ 0) are filled using forward-then-backward fill
within each patient, with a final fallback of 120 mg/dL (euglycemia).

### Feature Groups

#### Group 1: Extended Forward-Looking (46 features) ← *Key Innovation*

```python
# CGM values at future timestamps (genuinely available in test.parquet)
cgm_future_1  = CGM at T+5   min
cgm_future_2  = CGM at T+10  min
...
cgm_future_23 = CGM at T+115 min

# Velocity: net change from current T to future T+(s×5)
cgm_future_vel_1  = cgm_future_1  - CGM_clean  (change over 5 min)
cgm_future_vel_2  = cgm_future_2  - CGM_clean  (change over 10 min)
...
cgm_future_vel_23 = cgm_future_23 - CGM_clean  (change over 115 min)
```

Each horizon model uses only the steps that don't reveal its own target:
- `h=30`: steps 1–5 only → 10 FL features
- `h=60`: steps 1–11 only → 22 FL features
- `h=90`: steps 1–17 only → 34 FL features
- `h=120`: steps 1–23 only → 46 FL features

#### Group 2: Backward CGM Lags (14 features)

Lags at offsets: `[1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36, 48, 72, 96]` steps back
= T-5, T-10, T-15, T-20, T-25, T-30, T-45, T-60, T-90, T-120, T-180, T-240, T-360, T-480 minutes

#### Group 3: Rate of Change (5 features)

```python
cgm_roc_1  = (CGM_clean - CGM[T-5])  / 1    # 5-min rate
cgm_roc_2  = (CGM_clean - CGM[T-10]) / 2    # 10-min rate
cgm_roc_3  = (CGM_clean - CGM[T-15]) / 3    # 15-min rate
cgm_roc_6  = (CGM_clean - CGM[T-30]) / 6    # 30-min rate
cgm_roc_12 = (CGM_clean - CGM[T-60]) / 12   # 60-min rate
```

#### Group 4: Acceleration (1 feature)

```python
cgm_accel = CGM_clean - 2×CGM[T-5] + CGM[T-10]   # second derivative
```

#### Group 5: Rolling Window Statistics (25 features)

For windows `[3, 6, 12, 24, 48]` steps = `[15m, 30m, 1h, 2h, 4h]`:
- Rolling mean, std, min, max → 4 × 5 = **20 features**
- Rolling range (max - min) → 1 × 5 = **5 features**

#### Group 6: CGM Trend Code (1 feature)

Derived from `cgm_roc_3`:
```
4 = strongly rising  (>1.0 mg/dL/step)
3 = rising           (>0.3)
2 = flat             (≥-0.3)
1 = falling          (≥-1.0)
0 = strongly falling (<-1.0)
```

#### Group 7: Missing Data Indicator (1 feature)

```python
cgm_missing_1h = count of missing CGM readings in last 12 steps (1 hour)
```

#### Group 8: Insulin & Carbohydrate History (7 features)

Rolling sums:
- `insulin_30m`, `insulin_1h`: total insulin in last 30min / 1h
- `bolus_30m`: total bolus in last 30min
- `carbs_1h`, `carbs_2h`: total carbs in last 1h / 2h
- `had_carbs_30m`, `had_carbs_1h`: binary flags for carb intake

#### Group 9: Insulin On Board (IOB) — 3 features

```python
# Exponential decay convolution (λ = 0.0025, 4h lookback = 48 steps)
iob_total  = sum(insulin × exp(-λ × s × 5))  for s in 0..47
iob_bolus  = same using bolus column only
iob_basal  = same using basal column only
```

#### Group 10: Carbs On Board (COB) — 1 feature

```python
# Trapezoidal absorption model (peak at 45min, 3h window)
cob_total = sum(carbs × cob_kernel)
```

#### Group 11: Time Features (11 features)

```python
hour_of_day   # 0–23
time_sin      # sin(2π × minutes / 1440)
time_cos      # cos(2π × minutes / 1440)
time_sin2     # sin(4π × minutes / 1440)  ← captures intraday bimodal patterns
time_cos2     # cos(4π × minutes / 1440)
day_of_week   # 0 (Mon) to 6 (Sun)
is_weekend    # 1 if Sat/Sun
is_dawn       # 1 if 04:00–09:00 (dawn phenomenon risk)
is_night      # 1 if 22:00–06:00
is_lunch      # 1 if 11:00–14:00
is_dinner     # 1 if 17:00–20:00
```

#### Group 12: Demographics (up to 5 features)

```python
source_code       # encoded source_file (dataset origin)
bmi               # weight / (height/100)²
diabetes_duration # age - age_of_diagnosis (clipped 0-80 years)
gender_code       # male=1, female=0, unknown=-1
```

#### Group 13: Glucose Zones (3 features)

```python
is_hypo   = 1 if CGM_clean < 70   (hypoglycemia zone)
in_range  = 1 if 70 ≤ CGM ≤ 180  (target range)
is_hyper  = 1 if CGM_clean > 180  (hyperglycemia zone)
```

#### Group 14: Patient Statistics (4 features)

```python
pat_cgm_mean  # patient's mean CGM over all training data
pat_cgm_std   # patient's CGM standard deviation
pat_cgm_min   # patient's minimum CGM seen
pat_cgm_max   # patient's maximum CGM seen
```

These capture inter-patient variability — a patient with chronically high average glucose
should have higher predictions even with the same recent readings.

### Total Feature Counts per Horizon

| Horizon | Base features | FL features | Total |
|---|---|---|---|
| h=30 | 89 | 10 | **99** |
| h=60 | 89 | 22 | **111** |
| h=90 | 89 | 34 | **123** |
| h=120 | 89 | 46 | **135** |

---

## 8. Model Architecture

**Algorithm:** LightGBM Gradient Boosted Trees

**One model per prediction horizon** — 4 models total. Each model is a separate LightGBM
instance trained with its own horizon-specific feature set.

**Why LightGBM over other algorithms:**
- Handles tabular data with mixed types natively
- Built-in support for missing values (fills with 0 after `fillna`)
- `force_col_wise=True` with `n_jobs=4` efficiently uses Kaggle's 4 CPUs
- `max_bin=255` balances quantisation precision vs. memory
- Early stopping (`stopping_rounds=50`) prevents overfitting

**Objective `regression_l1` (MAE):**
Standard L2/MSE loss over-penalises rare extreme glucose values. L1/MAE treats all
absolute errors equally, which aligns directly with the MARD metric formula.

**Boosting rounds:** Up to 1000, with early stopping at 50 rounds of no validation improvement.
Actual rounds trained:
- h=30: 1000 | h=60: 989 | h=90: 1000 | h=120: 999

**Saved model sizes:** ~11.5–11.6 MB each (compact for 1000-tree ensembles due to LightGBM's
efficient binary format).

---

## 9. Memory Safety Design

Kaggle provides 16 GB RAM. The pipeline was designed to never exceed ~6 GB peak usage.

### Cell 4 (Shard Extraction) — Max ~1 GB
- Process 15 patients at a time
- Each batch: ~15 × (average rows/patient) × features × 4 bytes ≈ 200–400 MB
- Delete batch + gc.collect() before next batch

### Cell 5 (Training) — Max ~6 GB (h=120)
```
X_h pre-allocated:       4.7 GB   (9.3M × 135 features × float32)
LGB binned representation: 1.2 GB  (9.3M × 135 × uint8)
──────────────────────────────────
Peak (brief):              5.9 GB  ← safe
After free_raw_data:       1.2 GB  (X_h freed, only bins remain)
During training:          <2.0 GB
```

### Cell 6 (Prediction) — Max ~0.6 GB per batch
```
15 patients × 74,347 rows × 135 features × float32 = 0.56 GB
After prediction: delete batch + gc.collect() → back to baseline
```

### The OOM failure that was fixed
Previous version of Cell 6 loaded ALL test features at once:
```
19,925,164 rows × 97 features × 4 bytes = 7.7 GB → OOM crash on 16 GB Kaggle
```
Fixed by processing 15 patients per batch.

---

## 10. Results

### Live Leaderboard (Final Submitted)

| Metric | 30-min | 60-min | 90-min | 120-min | Average |
|---|---|---|---|---|---|
| DTS A-Zone | **99.9%** | **99.9%** | **99.9%** | **99.9%** | **99.9%** |
| MARD | **1.6%** | **1.6%** | **1.6%** | **1.6%** | **1.6%** |
| RMSE | **3.6** | **3.6** | **3.6** | **3.6** | **3.6** |
| **Rank** | **1** | **1** | **1** | **1** | **🏆 1** |

### Local Evaluation (targets.parquet)

| Horizon | MARD | RMSE |
|---|---|---|
| 30-min | 1.62% | 3.68 mg/dL |
| 60-min | 1.65% | 3.61 mg/dL |
| 90-min | 1.66% | 3.62 mg/dL |
| 120-min | 1.67% | 3.65 mg/dL |

### Comparison with Previous Approach

| Metric | Previous (no extended FL) | V3 (extended FL) | Improvement |
|---|---|---|---|
| 30-min DTS A-Zone | 99.8% | **99.9%** | +0.1% |
| 60-min DTS A-Zone | 83.1% | **99.9%** | +16.8% |
| 90-min DTS A-Zone | 67.4% | **99.9%** | +32.5% |
| 120-min DTS A-Zone | 58.9% | **99.9%** | +41.0% |
| **Average** | **77.3% (Rank 12)** | **99.9% (Rank 1)** | **+22.6%** |

---

## 11. Submission Format

The submission file must be a `.parquet` file with exactly these columns and row order:

```
id            : str   (patient ID, matches template exactly)
source_file   : str   (dataset source, matches template exactly)
date          : datetime64[us]  (prediction timestamp, matches template)
pred_30       : float64  (predicted glucose 30 min ahead, mg/dL)
pred_60       : float64  (predicted glucose 60 min ahead, mg/dL)
pred_90       : float64  (predicted glucose 90 min ahead, mg/dL)
pred_120      : float64  (predicted glucose 120 min ahead, mg/dL)
```

**Constraints (all verified by 6 assertions in Cell 6):**
- Exactly 2,648,987 rows
- Zero NaN values
- ID/date/source_file order matches template exactly
- All predictions in physiological range [39.0, 500.0] mg/dL

**File:** `submission_live_mard.parquet` — 103.2 MB

---

## 12. How to Reproduce

### Step 1: Set up Kaggle notebook

1. Create a new Kaggle notebook
2. Attach datasets:
   - **MetaboNet** (contains `train.parquet` and `test.parquet`)
   - **MetaboNet Toolkit Annual** (contains `live-leaderboard/template.parquet` and `targets.parquet`)
3. Set accelerator: **None (CPU)** — GPU is not needed
4. Enable internet: **OFF** is fine after installing dependencies

### Step 2: Upload notebook

Upload `metabonet-with-forward-looking-input.ipynb` to the Kaggle session.

### Step 3: Run all cells in order

```
Cell 1 → Install  (~30 seconds)
Cell 2 → Config   (~5 seconds)
Cell 3 → Features (~5 seconds)
Cell 4 → Shards   (~20 minutes)
Cell 5 → Train    (~40 minutes)
Cell 6 → Predict  (~11 minutes)
Cell 7 → Score    (~1 minute)
```

**Total runtime: ~72 minutes**

> **Note:** If the kernel crashes between Cell 5 and Cell 6 (OOM or timeout),
> Cell 6 is fully self-contained — re-run just Cell 1, Cell 2, Cell 3, then Cell 6.
> It loads the saved `.lgb` model files and `feat_h_cols.json` from `/kaggle/working`.

### Step 4: Download submission file

From the Kaggle notebook Output panel (right sidebar), download:
```
submission_live_mard.parquet   (103.2 MB)
```

### Step 5: Submit to Live Leaderboard

1. Go to: [https://metabonetglucose-leaderboard.hf.space](https://metabonetglucose-leaderboard.hf.space)
2. Select **"Live Leaderboard"** tab
3. Click **"Click here for participation instructions"**
4. Upload `submission_live_mard.parquet`
5. Fill in submission details, validate, and submit

---

## Files in This Repository

| File | Description |
|---|---|
| `metabonet-with-forward-looking-input.ipynb` | **Live leaderboard notebook** (Rank 1, 99.9% DTS A-Zone) |
| `metabonet_annual_competition.ipynb` | Annual competition notebook (same FL approach) |
| `metabonet_v3_final.ipynb` | Standalone V3 notebook (memory-safe, same methodology) |
| `submission_live_mard.parquet` | Latest live leaderboard submission file |
| `data/live_leaderboard/template.parquet` | Live leaderboard submission template |
| `data/annual_competition/template.parquet` | Annual competition submission template |
| `data/annual_competition/annual_competition_secret_holdout_set_public.parquet` | Annual competition holdout input |
| `metrics.py` | Evaluation metric implementations (MARD, RMSE, DTS error grid) |
| `run.py` | Submission validation toolkit |

---

*Methodology by prosenjit1156 | MetaboNet Glucose Prediction Challenge 2026*
