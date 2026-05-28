import psycopg2
import psutil
import json
import numpy as np
import pandas as pd
import joblib
import sqlglot

# ════════════════════════════════════════════════════════════════
# LOAD MODEL AND SCALER ONCE AT STARTUP
# ════════════════════════════════════════════════════════════════

model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURE_COLS = [
    "plan_node_count", "plan_depth", "plan_total_cost", "plan_avg_cost",
    "plan_total_rows", "plan_max_rows", "row_estimate_ratio",
    "plan_join_count", "plan_seq_scan_count", "plan_index_scan_count",
    "sem_join_count", "sem_condition_count", "sem_subquery_count",
    "sem_table_count", "sem_has_group_by", "sem_has_order_by",
    "sem_has_having", "sem_has_limit",
    "sys_cpu_percent", "sys_memory_percent",
    "sys_disk_read_bytes", "sys_disk_write_bytes",
]

DB_CONFIG = {
    "host": "localhost", "database": "tpch",
    "user": "postgres", "password": "Arivu1123", "port": "5432"
}


# ════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (same logic as Phase 3)
# ════════════════════════════════════════════════════════════════

def walk_plan(node, acc):
    acc["node_types"].append(node.get("Node Type", ""))
    acc["total_costs"].append(node.get("Total Cost", 0))
    acc["plan_rows"].append(node.get("Plan Rows", 0))
    acc["actual_rows"].append(node.get("Actual Rows", 0))
    acc["depth"] += 1
    if "Join" in node.get("Node Type", ""):
        acc["join_count"] += 1
    if node.get("Node Type") == "Seq Scan":
        acc["seq_scan_count"] += 1
    elif "Index" in node.get("Node Type", ""):
        acc["index_scan_count"] += 1
    for child in node.get("Plans", []):
        walk_plan(child, acc)

def extract_plan_features(plan_node):
    acc = {"node_types": [], "total_costs": [], "plan_rows": [],
           "actual_rows": [], "depth": 0, "join_count": 0,
           "seq_scan_count": 0, "index_scan_count": 0}
    walk_plan(plan_node, acc)
    return {
        "plan_node_count":       len(acc["node_types"]),
        "plan_depth":            acc["depth"],
        "plan_total_cost":       max(acc["total_costs"]) if acc["total_costs"] else 0,
        "plan_avg_cost":         np.mean(acc["total_costs"]) if acc["total_costs"] else 0,
        "plan_total_rows":       sum(acc["plan_rows"]),
        "plan_max_rows":         max(acc["plan_rows"]) if acc["plan_rows"] else 0,
        "row_estimate_ratio":    sum(acc["actual_rows"]) / max(sum(acc["plan_rows"]), 1),
        "plan_join_count":       acc["join_count"],
        "plan_seq_scan_count":   acc["seq_scan_count"],
        "plan_index_scan_count": acc["index_scan_count"],
    }

def extract_semantic_features(sql):
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
        return {
            "sem_join_count":      len(list(parsed.find_all(sqlglot.exp.Join))),
            "sem_condition_count": len(list(parsed.find_all(sqlglot.exp.Condition))),
            "sem_subquery_count":  len(list(parsed.find_all(sqlglot.exp.Subquery))),
            "sem_table_count":     len(list(parsed.find_all(sqlglot.exp.Table))),
            "sem_has_group_by":    1 if parsed.find(sqlglot.exp.Group)  else 0,
            "sem_has_order_by":    1 if parsed.find(sqlglot.exp.Order)  else 0,
            "sem_has_having":      1 if parsed.find(sqlglot.exp.Having) else 0,
            "sem_has_limit":       1 if parsed.find(sqlglot.exp.Limit)  else 0,
        }
    except:
        return {k: 0 for k in [
            "sem_join_count", "sem_condition_count", "sem_subquery_count",
            "sem_table_count", "sem_has_group_by", "sem_has_order_by",
            "sem_has_having", "sem_has_limit"]}


# ════════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ════════════════════════════════════════════════════════════════

def predict_query_time(sql: str, verbose: bool = True) -> dict:
    """
    Takes any SQL query string.
    Returns predicted execution time in milliseconds.
    Also returns actual execution time so you can compare.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    try:
        # 1. Snapshot system before
        io_before = psutil.disk_io_counters()
        sys_features = {
            "sys_cpu_percent":      psutil.cpu_percent(interval=0.1),
            "sys_memory_percent":   psutil.virtual_memory().percent,
            "sys_disk_read_bytes":  0,
            "sys_disk_write_bytes": 0,
        }

        # 2. Run EXPLAIN ANALYZE to get plan + actual time
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan_json  = cur.fetchone()[0][0]
        actual_ms  = plan_json["Execution Time"]
        plan_node  = plan_json["Plan"]

        # 3. Update disk metrics after execution
        io_after = psutil.disk_io_counters()
        sys_features["sys_disk_read_bytes"]  = io_after.read_bytes  - io_before.read_bytes
        sys_features["sys_disk_write_bytes"] = io_after.write_bytes - io_before.write_bytes

        # 4. Extract all features
        plan_feats = extract_plan_features(plan_node)
        sem_feats  = extract_semantic_features(sql)

        all_features = {}
        all_features.update(plan_feats)
        all_features.update(sem_feats)
        all_features.update(sys_features)

        # 5. Build feature vector in correct column order
        feature_vector = pd.DataFrame([all_features])[FEATURE_COLS]

        # 6. Scale using the saved scaler
        feature_scaled = scaler.transform(feature_vector)

        # 7. Predict (in log scale, convert back to ms)
        pred_log = model.predict(feature_scaled)[0]
        pred_ms  = np.expm1(pred_log)

        error_ms  = pred_ms - actual_ms
        error_pct = abs(error_ms / max(actual_ms, 1)) * 100

        result = {
            "predicted_ms": round(float(pred_ms), 2),
            "actual_ms":    round(float(actual_ms), 2),
            "error_ms":     round(float(error_ms), 2),
            "error_pct":    round(float(error_pct), 2),
            "features":     all_features,
        }

        if verbose:
            print(f"\n{'='*48}")
            print(f"  SQL (truncated): {sql.strip()[:60]}...")
            print(f"{'='*48}")
            print(f"  Predicted time : {pred_ms:.2f} ms")
            print(f"  Actual time    : {actual_ms:.2f} ms")
            print(f"  Error          : {error_ms:+.2f} ms  ({error_pct:.1f}%)")
            print(f"  Plan depth     : {plan_feats['plan_depth']}")
            print(f"  Join count     : {plan_feats['plan_join_count']}")
            print(f"  Seq scans      : {plan_feats['plan_seq_scan_count']}")
            print(f"{'='*48}\n")

        return result

    finally:
        cur.close()
        conn.close()


# ════════════════════════════════════════════════════════════════
# TEST THE PREDICTOR ON 5 EXAMPLE QUERIES
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    test_queries = [
        # Simple
        "SELECT COUNT(*) FROM orders",

        # Medium
        """SELECT c.c_mktsegment, SUM(o.o_totalprice)
           FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey
           GROUP BY c.c_mktsegment ORDER BY 2 DESC""",

        # Complex
        """SELECT n.n_name, COUNT(*) as cnt
           FROM customer c
           JOIN orders o    ON c.c_custkey  = o.o_custkey
           JOIN nation n    ON c.c_nationkey = n.n_nationkey
           WHERE o.o_orderdate BETWEEN '1994-01-01' AND '1997-12-31'
           GROUP BY n.n_name ORDER BY cnt DESC""",

        # With subquery
        """SELECT c_name FROM customer
           WHERE c_custkey IN (
               SELECT o_custkey FROM orders
               WHERE o_totalprice > (SELECT AVG(o_totalprice) FROM orders)
           ) LIMIT 20""",

        # Heavy aggregation
        """SELECT l_returnflag, l_linestatus,
                  SUM(l_quantity), SUM(l_extendedprice),
                  AVG(l_discount), COUNT(*)
           FROM lineitem
           WHERE l_shipdate <= '1998-09-01'
           GROUP BY l_returnflag, l_linestatus
           ORDER BY l_returnflag, l_linestatus""",
    ]

    print("\nRunning predictions on 5 test queries...\n")
    results = []
    for sql in test_queries:
        r = predict_query_time(sql, verbose=True)
        results.append(r)

    # Final summary table
    print("\n" + "=" * 60)
    print(f"  {'Query':<8} {'Predicted':>12} {'Actual':>12} {'Error %':>10}")
    print("  " + "─" * 46)
    for i, r in enumerate(results, 1):
        print(f"  Q{i:<7} {r['predicted_ms']:>10.2f}ms "
              f"{r['actual_ms']:>10.2f}ms "
              f"{r['error_pct']:>9.1f}%")
    print("=" * 60)

    avg_err = np.mean([r["error_pct"] for r in results])
    print(f"\n  Average prediction error: {avg_err:.1f}%")
    print("\n  Predictor is working! Phase 6 complete.")