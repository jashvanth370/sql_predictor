import psycopg2
import psutil
import json
import csv
import time
import os

DB_CONFIG = {
    "host":     "localhost",
    "database": "tpch",
    "user":     "postgres",
    "password": "Arivu1123",   # ← change this
    "port":     "5432"
}

# ─── 10 varied queries — different complexity levels ─────────────────────
QUERIES = [
    ("Q_simple_count",
     "SELECT COUNT(*) FROM orders"),

    ("Q_simple_filter",
     "SELECT COUNT(*) FROM orders WHERE o_totalprice > 100000"),

    ("Q_simple_date",
     "SELECT COUNT(*) FROM orders WHERE o_orderdate >= '1995-01-01'"),

    ("Q_one_join",
     """SELECT c.c_name, COUNT(o.o_orderkey) as num_orders
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_name
        ORDER BY num_orders DESC LIMIT 20"""),

    ("Q_two_joins",
     """SELECT c.c_mktsegment, SUM(o.o_totalprice) as total
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        JOIN lineitem l ON o.o_orderkey = l.l_orderkey
        GROUP BY c.c_mktsegment
        ORDER BY total DESC"""),

    ("Q_three_joins",
     """SELECT n.n_name, COUNT(*) as orders
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        JOIN nation n ON c.c_nationkey = n.n_nationkey
        WHERE o.o_orderdate BETWEEN '1994-01-01' AND '1997-12-31'
        GROUP BY n.n_name
        ORDER BY orders DESC"""),

    ("Q_aggregation",
     """SELECT l_returnflag, l_linestatus,
               SUM(l_quantity) as sum_qty,
               SUM(l_extendedprice) as sum_price,
               AVG(l_discount) as avg_discount,
               COUNT(*) as count_order
        FROM lineitem
        WHERE l_shipdate <= '1998-09-01'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus"""),

    ("Q_subquery",
     """SELECT c_name, c_acctbal
        FROM customer
        WHERE c_custkey IN (
            SELECT o_custkey FROM orders
            WHERE o_totalprice > (
                SELECT AVG(o_totalprice) FROM orders
            )
        )
        LIMIT 50"""),

    ("Q_four_joins",
     """SELECT s.s_name, n.n_name, SUM(l.l_extendedprice) as revenue
        FROM supplier s
        JOIN nation n ON s.s_nationkey = n.n_nationkey
        JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
        JOIN lineitem l ON ps.ps_partkey = l.l_partkey
                       AND ps.ps_suppkey = l.l_suppkey
        GROUP BY s.s_name, n.n_name
        ORDER BY revenue DESC
        LIMIT 20"""),

    ("Q_complex_filter",
     """SELECT o.o_orderpriority, COUNT(*) as order_count
        FROM orders o
        WHERE o.o_orderdate >= '1993-07-01'
          AND o.o_orderdate < '1993-10-01'
          AND EXISTS (
              SELECT 1 FROM lineitem l
              WHERE l.l_orderkey = o.o_orderkey
                AND l.l_commitdate < l.l_receiptdate
          )
        GROUP BY o.o_orderpriority
        ORDER BY o.o_orderpriority"""),
]

def get_system_before():
    io = psutil.disk_io_counters()
    return {
        "cpu":        psutil.cpu_percent(interval=0.1),
        "memory":     psutil.virtual_memory().percent,
        "disk_read":  io.read_bytes,
        "disk_write": io.write_bytes,
    }

def get_system_after(before):
    io = psutil.disk_io_counters()
    return {
        "cpu_percent":      before["cpu"],
        "memory_percent":   before["memory"],
        "disk_read_bytes":  io.read_bytes  - before["disk_read"],
        "disk_write_bytes": io.write_bytes - before["disk_write"],
    }

def collect_one(cur, query_name, sql, run_number):
    try:
        sys_before = get_system_before()

        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan = cur.fetchone()[0][0]

        sys_after = get_system_after(sys_before)

        exec_time = plan["Execution Time"]
        plan_node = plan["Plan"]

        print(f"    run {run_number}: {exec_time:.2f} ms")

        return {
            "query_name":       query_name,
            "run_number":       run_number,
            "execution_time":   exec_time,           # TARGET variable
            "planning_time":    plan["Planning Time"],
            "plan_json":        json.dumps(plan_node),
            "sql":              sql.strip(),
            "cpu_percent":      sys_after["cpu_percent"],
            "memory_percent":   sys_after["memory_percent"],
            "disk_read_bytes":  sys_after["disk_read_bytes"],
            "disk_write_bytes": sys_after["disk_write_bytes"],
        }
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def collect_dataset(runs_per_query=10):
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    print("Connected to PostgreSQL tpch database.\n")

    os.makedirs("data", exist_ok=True)
    output_file = "data/raw_data.csv"

    all_rows = []
    total = len(QUERIES) * runs_per_query

    for i, (name, sql) in enumerate(QUERIES):
        print(f"[{i+1}/{len(QUERIES)}] {name}")
        for run in range(1, runs_per_query+1):
            row = collect_one(cur, name, sql, run)
            if row:
                all_rows.append(row)
            time.sleep(0.3)

    # Save to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*45}")
    print(f"Collected {len(all_rows)} rows from {len(QUERIES)} queries")
    print(f"Saved to: {output_file}")
    print(f"{'='*45}")

    # Quick summary
    import pandas as pd
    df = pd.read_csv(output_file)
    print("\nExecution time summary (ms):")
    print(df.groupby("query_name")["execution_time"]
            .mean().round(2).to_string())

    cur.close()
    conn.close()

if __name__ == "__main__":
    collect_dataset(runs_per_query=10)