import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# LOAD PREPROCESSED DATA
# ════════════════════════════════════════════════════════════════

print("Loading preprocessed data...")

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()   # log scale
y_test  = pd.read_csv("data/y_test.csv").squeeze()    # log scale
y_test_raw = pd.read_csv("data/y_test_raw.csv").squeeze()  # original ms

print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Features         : {len(X_train.columns)}")
print()


# ════════════════════════════════════════════════════════════════
# HELPER: EVALUATE A TRAINED MODEL
# ════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test_log, y_test_raw, model_name):
    """
    Evaluates a trained model and prints a full report.
    We predict in log scale, then convert back to ms for reporting.
    """
    # Predict (still in log scale)
    y_pred_log = model.predict(X_test)

    # Convert predictions back to milliseconds
    # We used log1p to transform, so we use expm1 to reverse it
    y_pred_ms = np.expm1(y_pred_log)
    y_true_ms = y_test_raw.values

    # Calculate metrics in original ms scale (easier to understand)
    mae  = mean_absolute_error(y_true_ms, y_pred_ms)
    mse  = mean_squared_error(y_true_ms, y_pred_ms)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true_ms, y_pred_ms)

    # Also calculate R² in log scale (used for cross-validation)
    r2_log = r2_score(y_test_log, y_pred_log)

    print(f"\n  --- {model_name} Test Results ---")
    print(f"  MAE  (mean absolute error) : {mae:.2f} ms")
    print(f"  RMSE (root mean sq error)  : {rmse:.2f} ms")
    print(f"  R²   (log scale)           : {r2_log:.4f}")
    print(f"  R²   (ms scale)            : {r2:.4f}")
    print(f"  (R² of 1.0 = perfect, 0.0 = no better than guessing mean)")

    # Show a few example predictions vs actual
    print(f"\n  Sample predictions vs actual:")
    print(f"  {'Actual (ms)':>12}  {'Predicted (ms)':>14}  {'Error':>10}")
    print(f"  {'-'*40}")
    for actual, predicted in zip(y_true_ms[:6], y_pred_ms[:6]):
        error = predicted - actual
        print(f"  {actual:>12.2f}  {predicted:>14.2f}  {error:>+10.2f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "R2_log": r2_log}


# ════════════════════════════════════════════════════════════════
# MODEL 1 — RANDOM FOREST
# ════════════════════════════════════════════════════════════════

print("=" * 52)
print("Training Model 1: Random Forest")
print("=" * 52)

# Step 1: Start with reasonable defaults and cross-validate
print("\nStep 1: Cross-validation with default settings...")
rf_default = RandomForestRegressor(
    n_estimators=100,   # 100 trees
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

cv_scores = cross_val_score(
    rf_default, X_train, y_train,
    cv=5,               # 5-fold: splits training data into 5 chunks
    scoring="r2",       # measure R² on each fold
    n_jobs=-1
)
print(f"  5-fold CV R² scores : {cv_scores.round(3)}")
print(f"  Mean CV R²          : {cv_scores.mean():.4f}")
print(f"  Std  CV R²          : {cv_scores.std():.4f}")

# Step 2: Hyperparameter tuning with GridSearchCV
# GridSearchCV tries every combination and picks the best one
print("\nStep 2: Hyperparameter tuning (this may take ~30 seconds)...")

rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth":    [None, 10, 20],
    "min_samples_split": [2, 5],
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0
)

start = time.time()
rf_grid.fit(X_train, y_train)
elapsed = time.time() - start

print(f"  Tuning complete in {elapsed:.1f}s")
print(f"  Best parameters : {rf_grid.best_params_}")
print(f"  Best CV R²      : {rf_grid.best_score_:.4f}")

# Step 3: Train final model with best parameters
rf_best = rf_grid.best_estimator_
print("\nStep 3: Training final Random Forest with best params...")
rf_best.fit(X_train, y_train)
print("  Training complete.")

# Step 4: Evaluate on test set
rf_results = evaluate_model(rf_best, X_test, y_test, y_test_raw, "Random Forest")

# Step 5: Feature importance — which features matter most?
print("\n  Top 10 most important features:")
importances = pd.Series(rf_best.feature_importances_, index=X_train.columns)
top10 = importances.sort_values(ascending=False).head(10)
for feat, imp in top10.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:30s} {imp:.4f} {bar}")

# Save model
joblib.dump(rf_best, "models/rf_model.pkl")
print("\n  Saved: models/rf_model.pkl")


# ════════════════════════════════════════════════════════════════
# MODEL 2 — XGBOOST
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 52)
print("Training Model 2: XGBoost")
print("=" * 52)

# Step 1: Cross-validate with defaults
print("\nStep 1: Cross-validation with default settings...")
xgb_default = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    verbosity=0          # suppress XGBoost output
)

cv_scores_xgb = cross_val_score(
    xgb_default, X_train, y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
print(f"  5-fold CV R² scores : {cv_scores_xgb.round(3)}")
print(f"  Mean CV R²          : {cv_scores_xgb.mean():.4f}")
print(f"  Std  CV R²          : {cv_scores_xgb.std():.4f}")

# Step 2: Hyperparameter tuning
print("\nStep 2: Hyperparameter tuning (this may take ~30 seconds)...")

xgb_param_grid = {
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 5, 7],
}

xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    xgb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0
)

start = time.time()
xgb_grid.fit(X_train, y_train)
elapsed = time.time() - start

print(f"  Tuning complete in {elapsed:.1f}s")
print(f"  Best parameters : {xgb_grid.best_params_}")
print(f"  Best CV R²      : {xgb_grid.best_score_:.4f}")

# Step 3: Train final model
xgb_best = xgb_grid.best_estimator_
print("\nStep 3: Training final XGBoost with best params...")
xgb_best.fit(X_train, y_train)
print("  Training complete.")

# Step 4: Evaluate
xgb_results = evaluate_model(xgb_best, X_test, y_test, y_test_raw, "XGBoost")

# Step 5: Feature importance
print("\n  Top 10 most important features:")
xgb_importances = pd.Series(xgb_best.feature_importances_, index=X_train.columns)
top10_xgb = xgb_importances.sort_values(ascending=False).head(10)
for feat, imp in top10_xgb.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:30s} {imp:.4f} {bar}")

# Save model
joblib.dump(xgb_best, "models/xgb_model.pkl")
print("\n  Saved: models/xgb_model.pkl")


# ════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 52)
print("Final Model Comparison")
print("=" * 52)

print(f"\n  {'Metric':<20} {'Random Forest':>15} {'XGBoost':>15}")
print(f"  {'-'*52}")
print(f"  {'MAE (ms)':<20} {rf_results['MAE']:>15.2f} {xgb_results['MAE']:>15.2f}")
print(f"  {'RMSE (ms)':<20} {rf_results['RMSE']:>15.2f} {xgb_results['RMSE']:>15.2f}")
print(f"  {'R² (ms scale)':<20} {rf_results['R2']:>15.4f} {xgb_results['R2']:>15.4f}")
print(f"  {'R² (log scale)':<20} {rf_results['R2_log']:>15.4f} {xgb_results['R2_log']:>15.4f}")

# Pick the winner by R²
if xgb_results["R2"] >= rf_results["R2"]:
    winner = "XGBoost"
    winner_model = xgb_best
else:
    winner = "Random Forest"
    winner_model = rf_best

print(f"\n  Best model: {winner}")
joblib.dump(winner_model, "models/best_model.pkl")
print(f"  Saved as:   models/best_model.pkl")

print(f"""
{'='*52}
Phase 5 Complete!
{'='*52}
  models/rf_model.pkl    — Random Forest
  models/xgb_model.pkl   — XGBoost
  models/best_model.pkl  — Best of the two

  Ready for Phase 6 — Evaluation & Prediction!
{'='*52}
""")