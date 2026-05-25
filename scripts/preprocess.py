import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)


# ════════════════════════════════════════════════════════════════
# LOAD YOUR FEATURE DATASET
# ════════════════════════════════════════════════════════════════

print("Loading features.csv ...")
df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns.\n")


# ════════════════════════════════════════════════════════════════
# STEP 1 — HANDLE MISSING VALUES
# ════════════════════════════════════════════════════════════════
# Missing values crash the model. We fill numeric gaps with the
# median (not mean — median is safer when outliers exist).

print("Step 1: Handling missing values...")
before = df.isnull().sum().sum()

numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' with median={median_val:.3f}")

after = df.isnull().sum().sum()
print(f"  Missing values: {before} → {after}")
print("  Done.\n")


# ════════════════════════════════════════════════════════════════
# STEP 2 — LOG-TRANSFORM THE TARGET VARIABLE
# ════════════════════════════════════════════════════════════════
# Execution times are right-skewed — a few queries take 1000ms,
# most take under 50ms. This makes the model focus too much on
# slow outliers. Log-transform compresses the scale so the model
# treats all queries more equally.
#
# Example:
#   4ms   → log(4)   = 1.39
#   50ms  → log(50)  = 3.91
#   500ms → log(500) = 6.21   (gap is now proportional, not huge)

print("Step 2: Log-transforming execution_time...")
print(f"  Before — min: {df['execution_time'].min():.2f}ms  "
      f"max: {df['execution_time'].max():.2f}ms  "
      f"mean: {df['execution_time'].mean():.2f}ms")

# Add a tiny constant (1) to handle any zero values safely
df["log_execution_time"] = np.log1p(df["execution_time"])

print(f"  After  — min: {df['log_execution_time'].min():.3f}  "
      f"max: {df['log_execution_time'].max():.3f}  "
      f"mean: {df['log_execution_time'].mean():.3f}")
print("  Done.\n")


# ════════════════════════════════════════════════════════════════
# STEP 3 — DEFINE FEATURE COLUMNS
# ════════════════════════════════════════════════════════════════
# Separate features (X) from targets and labels.

FEATURE_COLS = [
    # Plan features
    "plan_node_count",
    "plan_depth",
    "plan_total_cost",
    "plan_avg_cost",
    "plan_total_rows",
    "plan_max_rows",
    "row_estimate_ratio",
    "plan_join_count",
    "plan_seq_scan_count",
    "plan_index_scan_count",
    # Semantic features
    "sem_join_count",
    "sem_condition_count",
    "sem_subquery_count",
    "sem_table_count",
    "sem_has_group_by",
    "sem_has_order_by",
    "sem_has_having",
    "sem_has_limit",
    # System features
    "sys_cpu_percent",
    "sys_memory_percent",
    "sys_disk_read_bytes",
    "sys_disk_write_bytes",
]

TARGET_RAW = "execution_time"       # original ms — used for final reporting
TARGET_LOG = "log_execution_time"   # log scale — used for training

print(f"Step 3: Feature columns defined — {len(FEATURE_COLS)} features total.")
print("  Done.\n")


# ════════════════════════════════════════════════════════════════
# STEP 4 — SCALE NUMERIC FEATURES
# ════════════════════════════════════════════════════════════════
# Features are on very different scales right now:
#   plan_total_cost  might be 50,000
#   sys_cpu_percent  might be 12
#   sem_join_count   might be 3
#
# StandardScaler brings everything to mean=0, std=1.
# This prevents large-valued features from dominating the model.
#
# IMPORTANT: We fit the scaler ONLY on training data, then apply
# it to test data. Never fit on test data — that would be cheating.

print("Step 4: Scaling numeric features with StandardScaler...")

X = df[FEATURE_COLS].copy()
y_log = df[TARGET_LOG].copy()
y_raw = df[TARGET_RAW].copy()

# Split FIRST, then scale — this is the correct order
X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.2,
    random_state=42     # random_state=42 means reproducible — same split every run
)

print(f"  Train size: {len(X_train)} rows")
print(f"  Test size:  {len(X_test)} rows")

# Fit scaler on TRAINING data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # apply (don't refit) on test

# Convert back to DataFrames so column names are preserved
X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS)

print("  Scaling complete.")
print(f"  Sample — plan_total_cost before: mean={X_train['plan_total_cost'].mean():.1f}")
print(f"           plan_total_cost after:  mean={X_train_scaled['plan_total_cost'].mean():.4f}")
print("  Done.\n")


# ════════════════════════════════════════════════════════════════
# STEP 5 — SANITY CHECKS
# ════════════════════════════════════════════════════════════════
# Before saving, verify everything looks correct.

print("Step 5: Running sanity checks...")

checks_passed = True

# Check 1: No missing values remain
missing = X_train_scaled.isnull().sum().sum()
if missing == 0:
    print("  No missing values in training features.")
else:
    print(f"  WARNING: {missing} missing values found!")
    checks_passed = False

# Check 2: Scaled features are centered near zero
means = X_train_scaled.mean()
if (means.abs() < 0.01).all():
    print("  All features centered near zero after scaling.")
else:
    print("  WARNING: Some features not centered near zero.")
    checks_passed = False

# Check 3: Target variable looks reasonable
print(f"  Target (log scale) — min: {y_train.min():.2f}  "
      f"max: {y_train.max():.2f}  mean: {y_train.mean():.2f}")

# Check 4: No data leakage — test set is separate
assert len(set(X_train.index) & set(X_test.index)) == 0, "DATA LEAKAGE DETECTED!"
print("  No data leakage — train and test indices are separate.")

if checks_passed:
    print("  All checks passed.\n")
else:
    print("  Some checks failed — review warnings above.\n")


# ════════════════════════════════════════════════════════════════
# STEP 6 — SAVE EVERYTHING
# ════════════════════════════════════════════════════════════════
# Save the processed datasets and the scaler object.
# The scaler must be saved because you need it later to scale
# new queries before prediction.

print("Step 6: Saving preprocessed data and scaler...")

X_train_scaled.to_csv("data/X_train.csv", index=False)
X_test_scaled.to_csv("data/X_test.csv",   index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)

y_train_raw.to_csv("data/y_train_raw.csv", index=False)
y_test_raw.to_csv("data/y_test_raw.csv",   index=False)

# Save scaler — needed to preprocess new queries at prediction time
joblib.dump(scaler, "models/scaler.pkl")

print("  Saved: data/X_train.csv")
print("  Saved: data/X_test.csv")
print("  Saved: data/y_train.csv  (log scale)")
print("  Saved: data/y_test.csv   (log scale)")
print("  Saved: data/y_train_raw.csv  (original ms)")
print("  Saved: data/y_test_raw.csv   (original ms)")
print("  Saved: models/scaler.pkl")


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

print(f"""
{'='*52}
Phase 4 Complete — Preprocessing Summary
{'='*52}
  Total samples  : {len(df)}
  Features       : {len(FEATURE_COLS)}
  Training rows  : {len(X_train_scaled)}
  Testing rows   : {len(X_test_scaled)}

  Target range (original ms):
    Min  : {y_raw.min():.2f} ms
    Max  : {y_raw.max():.2f} ms
    Mean : {y_raw.mean():.2f} ms

  Files saved to data/ and models/
  Ready for Phase 5 — Model Training!
{'='*52}
""")