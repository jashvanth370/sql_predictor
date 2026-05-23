import json
import pandas as pd
import numpy as np
import sqlglot

# ════════════════════════════════════════════════════════════════
# GROUP A: PLAN FEATURES
# ════════════════════════════════════════════════════════════════

# All node types PostgreSQL uses — we track counts of each
NODE_TYPES = [
    "Seq Scan", "Index Scan", "Index Only Scan",
    "Hash Join", "Nested Loop", "Merge Join",
    "Hash", "Sort", "Aggregate", "Group",
    "Limit", "Subquery Scan", "Materialize"
]

def walk_plan_tree(node, accumulator):
    """
    Recursively walks every node in the plan tree.
    For each node it visits, it adds data to the accumulator dict.
    Think of it like reading every page of a book, not just the cover.
    """
    # Count this node type
    node_type = node.get("Node Type", "Unknown")
    accumulator["node_types"].append(node_type)

    # Collect cost and row estimates from this node
    accumulator["total_costs"].append(node.get("Total Cost", 0))
    accumulator["plan_rows"].append(node.get("Plan Rows", 0))
    accumulator["actual_rows"].append(node.get("Actual Rows", 0))
    accumulator["depth"] += 1

    # Check if this node is a join
    if "Join" in node_type:
        accumulator["join_count"] += 1

    # Check if this is a sequential scan (slow) or index scan (fast)
    if node_type == "Seq Scan":
        accumulator["seq_scan_count"] += 1
    elif "Index" in node_type:
        accumulator["index_scan_count"] += 1

    # Recurse into child nodes
    for child in node.get("Plans", []):
        walk_plan_tree(child, accumulator)

def extract_plan_features(plan_json_str):
    """
    Takes the plan JSON string from your CSV and returns
    a flat dictionary of numeric features.
    """
    try:
        plan = json.loads(plan_json_str)
    except:
        return None

    # Initialize accumulator
    acc = {
        "node_types":       [],
        "total_costs":      [],
        "plan_rows":        [],
        "actual_rows":      [],
        "depth":            0,
        "join_count":       0,
        "seq_scan_count":   0,
        "index_scan_count": 0,
    }

    walk_plan_tree(plan, acc)

    # Now summarize the accumulator into flat numeric features
    features = {
        # Total number of operations in the plan
        "plan_node_count":    len(acc["node_types"]),

        # How deep is the plan tree (complex queries = deeper trees)
        "plan_depth":         acc["depth"],

        # Cost estimates (PostgreSQL's own guesses)
        "plan_total_cost":    max(acc["total_costs"]) if acc["total_costs"] else 0,
        "plan_avg_cost":      np.mean(acc["total_costs"]) if acc["total_costs"] else 0,

        # Row estimates
        "plan_total_rows":    sum(acc["plan_rows"]),
        "plan_max_rows":      max(acc["plan_rows"]) if acc["plan_rows"] else 0,

        # Row estimate accuracy (how wrong was PostgreSQL?)
        "row_estimate_ratio": (
            sum(acc["actual_rows"]) / max(sum(acc["plan_rows"]), 1)
        ),

        # Join information
        "plan_join_count":        acc["join_count"],

        # Scan types (index scans are faster than sequential)
        "plan_seq_scan_count":    acc["seq_scan_count"],
        "plan_index_scan_count":  acc["index_scan_count"],
    }

    return features   # 10 plan features


# ════════════════════════════════════════════════════════════════
# GROUP B: SEMANTIC FEATURES
# ════════════════════════════════════════════════════════════════

def extract_semantic_features(sql):
    """
    Parses the SQL text itself (not the plan) and counts
    structural elements that affect query complexity.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
    except:
        # If sqlglot can't parse it, return zeros
        return {
            "sem_join_count":      0,
            "sem_condition_count": 0,
            "sem_subquery_count":  0,
            "sem_table_count":     0,
            "sem_has_group_by":    0,
            "sem_has_order_by":    0,
            "sem_has_having":      0,
            "sem_has_limit":       0,
        }

    features = {
        # How many JOIN keywords appear?
        "sem_join_count": len(list(parsed.find_all(sqlglot.exp.Join))),

        # How many WHERE conditions (ANDs, ORs, comparisons)?
        "sem_condition_count": len(list(parsed.find_all(sqlglot.exp.Condition))),

        # Are there subqueries inside the main query?
        "sem_subquery_count": len(list(parsed.find_all(sqlglot.exp.Subquery))),

        # How many tables are referenced?
        "sem_table_count": len(list(parsed.find_all(sqlglot.exp.Table))),

        # Presence of clauses that add processing overhead
        "sem_has_group_by": 1 if parsed.find(sqlglot.exp.Group)   else 0,
        "sem_has_order_by": 1 if parsed.find(sqlglot.exp.Order)   else 0,
        "sem_has_having":   1 if parsed.find(sqlglot.exp.Having)  else 0,
        "sem_has_limit":    1 if parsed.find(sqlglot.exp.Limit)   else 0,
    }

    return features   # 8 semantic features


# ════════════════════════════════════════════════════════════════
# GROUP C: SYSTEM METRIC FEATURES
# ════════════════════════════════════════════════════════════════

def extract_system_features(row):
    """
    System metrics are already numbers in your CSV.
    This just picks them out cleanly and renames them.
    """
    return {
        "sys_cpu_percent":      row.get("cpu_percent", 0),
        "sys_memory_percent":   row.get("memory_percent", 0),
        "sys_disk_read_bytes":  row.get("disk_read_bytes", 0),
        "sys_disk_write_bytes": row.get("disk_write_bytes", 0),
    }   # 4 system features


# ════════════════════════════════════════════════════════════════
# COMBINE ALL THREE GROUPS
# ════════════════════════════════════════════════════════════════

def extract_all_features(row):
    """
    Takes one row from your raw_data.csv and returns
    a single dictionary with ALL 22 features combined.
    This becomes ONE row in your training dataset.
    """
    plan_feats   = extract_plan_features(row["plan_json"])
    sem_feats    = extract_semantic_features(row["sql"])
    sys_feats    = extract_system_features(row)

    if plan_feats is None:
        return None

    # Merge all three dictionaries into one
    all_features = {}
    all_features.update(plan_feats)   # 10 plan features
    all_features.update(sem_feats)    # 8  semantic features
    all_features.update(sys_feats)    # 4  system features

    # Add the target variable
    all_features["execution_time"] = row["execution_time"]

    # Keep query name for reference
    all_features["query_name"] = row["query_name"]

    return all_features   # 22 features + target + label


# ════════════════════════════════════════════════════════════════
# RUN FEATURE ENGINEERING ON YOUR FULL DATASET
# ════════════════════════════════════════════════════════════════

def build_feature_dataset(input_csv="data/raw_data.csv",
                          output_csv="data/features.csv"):
    import os
    os.makedirs("data", exist_ok=True)

    print("Loading raw data...")
    raw_df = pd.read_csv(input_csv)
    print(f"Loaded {len(raw_df)} rows.")

    print("Extracting features from each row...")
    feature_rows = []
    failed = 0

    for i, row in raw_df.iterrows():
        features = extract_all_features(row)
        if features:
            feature_rows.append(features)
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(raw_df)} rows...")

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_csv, index=False)

    print(f"\n{'='*50}")
    print(f"Feature extraction complete!")
    print(f"  Input rows:  {len(raw_df)}")
    print(f"  Output rows: {len(feature_df)}  (failed: {failed})")
    print(f"  Features:    {len(feature_df.columns) - 2} features + target + label")
    print(f"  Saved to:    {output_csv}")
    print(f"{'='*50}")

    return feature_df

if __name__ == "__main__":
    df = build_feature_dataset()

    print("\n--- Feature columns ---")
    for col in df.columns:
        print(f"  {col}")

    print("\n--- Sample values (first row) ---")
    print(df.iloc[0].to_string())

    print("\n--- Execution time stats ---")
    print(df["execution_time"].describe().round(3))