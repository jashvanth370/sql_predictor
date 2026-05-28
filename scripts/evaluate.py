import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD EVERYTHING
# ════════════════════════════════════════════════════════════════

print("Loading models and test data...")

rf_model  = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
scaler    = joblib.load("models/scaler.pkl")

X_test     = pd.read_csv("data/X_test.csv")
y_test_log = pd.read_csv("data/y_test.csv").squeeze()
y_test_raw = pd.read_csv("data/y_test_raw.csv").squeeze()
X_train    = pd.read_csv("data/X_train.csv")
y_train    = pd.read_csv("data/y_train.csv").squeeze()

print(f"  Test samples : {len(X_test)}")
print(f"  Features     : {len(X_test.columns)}\n")


# ════════════════════════════════════════════════════════════════
# STEP 2 — EVALUATE BOTH MODELS
# ════════════════════════════════════════════════════════════════

def full_evaluation(model, X_test, y_test_log, y_test_raw, name):
    """Full evaluation returning metrics and predictions in ms."""

    # Predict in log scale, convert back to ms
    y_pred_log = model.predict(X_test)
    y_pred_ms  = np.expm1(y_pred_log)
    y_true_ms  = y_test_raw.values

    mae  = mean_absolute_error(y_true_ms, y_pred_ms)
    mse  = mean_squared_error(y_true_ms, y_pred_ms)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true_ms, y_pred_ms)
    r2_log = r2_score(y_test_log, y_pred_log)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true_ms - y_pred_ms) / np.maximum(y_true_ms, 1))) * 100

    print(f"  {'─'*40}")
    print(f"  Model        : {name}")
    print(f"  {'─'*40}")
    print(f"  MAE          : {mae:.3f} ms")
    print(f"  RMSE         : {rmse:.3f} ms")
    print(f"  MAPE         : {mape:.2f}%")
    print(f"  R² (ms)      : {r2:.4f}")
    print(f"  R² (log)     : {r2_log:.4f}")
    print()

    return {
        "name": name, "mae": mae, "rmse": rmse,
        "mape": mape, "r2": r2, "r2_log": r2_log,
        "y_pred_ms": y_pred_ms, "y_true_ms": y_true_ms,
        "y_pred_log": y_pred_log,
    }

print("=" * 44)
print("Evaluation Results")
print("=" * 44 + "\n")

rf_eval  = full_evaluation(rf_model,  X_test, y_test_log, y_test_raw, "Random Forest")
xgb_eval = full_evaluation(xgb_model, X_test, y_test_log, y_test_raw, "XGBoost")


# ════════════════════════════════════════════════════════════════
# STEP 3 — VISUALIZE (4 charts in one figure)
# ════════════════════════════════════════════════════════════════

print("Generating evaluation charts...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Model Evaluation — SQL Query Execution Time Prediction",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

COLORS = {"Random Forest": "#2196F3", "XGBoost": "#FF5722"}

# ── Chart 1: Actual vs Predicted (both models) ──────────────────
ax1 = fig.add_subplot(gs[0, 0])

for eval_result in [rf_eval, xgb_eval]:
    ax1.scatter(
        eval_result["y_true_ms"],
        eval_result["y_pred_ms"],
        alpha=0.65, s=40,
        color=COLORS[eval_result["name"]],
        label=eval_result["name"]
    )

# Perfect prediction line (diagonal)
max_val = max(y_test_raw.max(), rf_eval["y_pred_ms"].max(), xgb_eval["y_pred_ms"].max())
ax1.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")

ax1.set_xlabel("Actual execution time (ms)")
ax1.set_ylabel("Predicted execution time (ms)")
ax1.set_title("Actual vs Predicted")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Annotation: points on the diagonal = perfect prediction
ax1.text(0.05, 0.92, "Points on dashed line = perfect prediction",
         transform=ax1.transAxes, fontsize=7.5, color="gray")


# ── Chart 2: Error Distribution ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

for eval_result in [rf_eval, xgb_eval]:
    errors = eval_result["y_pred_ms"] - eval_result["y_true_ms"]
    ax2.hist(errors, bins=15, alpha=0.6,
             color=COLORS[eval_result["name"]],
             label=eval_result["name"],
             edgecolor="white", linewidth=0.5)

ax2.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax2.set_xlabel("Prediction error (ms)  [predicted − actual]")
ax2.set_ylabel("Count")
ax2.set_title("Error Distribution")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.92, "Ideal: tall spike centred at 0",
         transform=ax2.transAxes, fontsize=7.5, color="gray")


# ── Chart 3: Feature Importance (top 12) ─────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

feature_names = X_test.columns.tolist()

# Use XGBoost importances (or switch to rf_model if preferred)
importances  = xgb_model.feature_importances_
feat_series  = pd.Series(importances, index=feature_names).sort_values()
top12        = feat_series.tail(12)

colors_imp = ["#FF5722" if v > top12.median() else "#FFCCBC" for v in top12.values]
bars = ax3.barh(top12.index, top12.values, color=colors_imp, edgecolor="white")

ax3.set_xlabel("Importance score")
ax3.set_title("Top 12 Feature Importances (XGBoost)")
ax3.grid(True, alpha=0.3, axis="x")

# Value labels on bars
for bar, val in zip(bars, top12.values):
    ax3.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=7.5)


# ── Chart 4: Metrics Comparison Bar Chart ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

metrics  = ["MAE (ms)", "RMSE (ms)", "R² × 100", "MAPE (%)"]
rf_vals  = [rf_eval["mae"],  rf_eval["rmse"],  rf_eval["r2"]  * 100, rf_eval["mape"]]
xgb_vals = [xgb_eval["mae"], xgb_eval["rmse"], xgb_eval["r2"] * 100, xgb_eval["mape"]]

x      = np.arange(len(metrics))
width  = 0.35

bars_rf  = ax4.bar(x - width/2, rf_vals,  width, label="Random Forest",
                   color=COLORS["Random Forest"], alpha=0.85, edgecolor="white")
bars_xgb = ax4.bar(x + width/2, xgb_vals, width, label="XGBoost",
                   color=COLORS["XGBoost"], alpha=0.85, edgecolor="white")

# Value labels on top of each bar
for bar in list(bars_rf) + list(bars_xgb):
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
             f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=8.5)
ax4.set_title("Metrics Comparison")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")
ax4.text(0.05, 0.92, "R² × 100: higher = better. Others: lower = better",
         transform=ax4.transAxes, fontsize=7.5, color="gray")

plt.savefig("outputs/evaluation_charts.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.show()
print("  Saved: outputs/evaluation_charts.png\n")


# ════════════════════════════════════════════════════════════════
# STEP 4 — COMPARE AGAINST POSTGRESQL BASELINE
# ════════════════════════════════════════════════════════════════
# PostgreSQL's own cost estimate is in plan_total_cost.
# We denormalise it back from the scaler so we can compare.

print("Comparing against PostgreSQL baseline...")

# Get the original (unscaled) plan_total_cost from raw features
features_raw = pd.read_csv("data/features.csv")
test_indices = y_test_raw.index
pg_costs_raw = features_raw.loc[test_indices, "plan_total_cost"].values

# PostgreSQL's cost is in arbitrary units — scale it to ms range
# using a linear fit so comparison is fair
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(pg_costs_raw.reshape(-1, 1), y_test_raw.values)
pg_pred_ms = lr.predict(pg_costs_raw.reshape(-1, 1))

pg_mae  = mean_absolute_error(y_test_raw.values, pg_pred_ms)
pg_rmse = np.sqrt(mean_squared_error(y_test_raw.values, pg_pred_ms))
pg_r2   = r2_score(y_test_raw.values, pg_pred_ms)

print(f"\n  {'─'*44}")
print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'─'*44}")
print(f"  {'PostgreSQL baseline':<22} {pg_mae:>8.2f} {pg_rmse:>8.2f} {pg_r2:>8.4f}")
print(f"  {'Random Forest':<22} {rf_eval['mae']:>8.2f} {rf_eval['rmse']:>8.2f} {rf_eval['r2']:>8.4f}")
print(f"  {'XGBoost':<22} {xgb_eval['mae']:>8.2f} {xgb_eval['rmse']:>8.2f} {xgb_eval['r2']:>8.4f}")
print(f"  {'─'*44}")

# Improvement over baseline
best_mae = min(rf_eval["mae"], xgb_eval["mae"])
improvement = ((pg_mae - best_mae) / pg_mae) * 100
print(f"\n  Your best model improves MAE by {improvement:.1f}% over PostgreSQL baseline.")