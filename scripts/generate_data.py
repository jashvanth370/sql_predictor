import psycopg2
import random
from faker import Faker
from datetime import date, timedelta
import time

fake = Faker()
random.seed(42)

DB_CONFIG = {
    "host":     "localhost",
    "database": "tpch",
    "user":     "postgres",
    "password": "Arivu1123",   # ← change this
    "port":     "5432"
}

# ─── Scale settings (increase for more data) ────────────────────────────
NUM_REGIONS    = 5
NUM_NATIONS    = 25
NUM_SUPPLIERS  = 500
NUM_CUSTOMERS  = 2000
NUM_PARTS      = 1000
NUM_ORDERS     = 5000      # increase to 20000 for more variety
NUM_LINEITEMS  = 4         # avg lineitems per order

REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
SHIP_MODES = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"]
CONTAINERS = ["SM BOX", "SM CASE", "SM PKG", "LG BOX", "LG CASE", "MED BAG"]

def random_date(start_year=1993, end_year=1998):
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Clear existing data (in correct order to respect foreign keys)
    print("Clearing old data...")
    cur.execute("""
        TRUNCATE lineitem, orders, partsupp, part,
                 customer, supplier, nation, region
        RESTART IDENTITY CASCADE;
    """)
    conn.commit()

    # ── Region ──────────────────────────────────────────────────────────
    print("Loading regions...")
    for i, name in enumerate(REGIONS):
        cur.execute(
            "INSERT INTO region VALUES (%s,%s,%s)",
            (i+1, name, fake.sentence())
        )
    conn.commit()

    # ── Nation ──────────────────────────────────────────────────────────
    print("Loading nations...")
    nation_names = [
        "ALGERIA","ARGENTINA","BRAZIL","CANADA","EGYPT",
        "ETHIOPIA","FRANCE","GERMANY","INDIA","INDONESIA",
        "IRAN","IRAQ","JAPAN","JORDAN","KENYA",
        "MOROCCO","MOZAMBIQUE","PERU","CHINA","ROMANIA",
        "SAUDI ARABIA","VIETNAM","RUSSIA","UNITED KINGDOM","UNITED STATES"
    ]
    for i, name in enumerate(nation_names):
        cur.execute(
            "INSERT INTO nation VALUES (%s,%s,%s,%s)",
            (i+1, name, (i % NUM_REGIONS)+1, fake.sentence())
        )
    conn.commit()

    # ── Supplier ─────────────────────────────────────────────────────────
    print(f"Loading {NUM_SUPPLIERS} suppliers...")
    for i in range(1, NUM_SUPPLIERS+1):
        cur.execute(
            "INSERT INTO supplier VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                i,
                f"Supplier#{i:09d}",
                fake.address()[:40],
                random.randint(1, NUM_NATIONS),
                fake.phone_number()[:15],
                round(random.uniform(-999, 9999), 2),
                fake.sentence()[:101]
            )
        )
    conn.commit()

    # ── Customer ─────────────────────────────────────────────────────────
    print(f"Loading {NUM_CUSTOMERS} customers...")
    for i in range(1, NUM_CUSTOMERS+1):
        cur.execute(
            "INSERT INTO customer VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                i,
                f"Customer#{i:09d}",
                fake.address()[:40],
                random.randint(1, NUM_NATIONS),
                fake.phone_number()[:15],
                round(random.uniform(-999, 9999), 2),
                random.choice(SEGMENTS),
                fake.sentence()[:117]
            )
        )
    conn.commit()

    # ── Part ─────────────────────────────────────────────────────────────
    print(f"Loading {NUM_PARTS} parts...")
    for i in range(1, NUM_PARTS+1):
        cur.execute(
            "INSERT INTO part VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                i,
                fake.word()[:55],
                f"Manufacturer#{random.randint(1,5)}",
                f"Brand#{random.randint(1,5)}{random.randint(1,5)}",
                fake.word()[:25],
                random.randint(1, 50),
                random.choice(CONTAINERS),
                round(random.uniform(100, 2000), 2),
                fake.sentence()[:23]
            )
        )
    conn.commit()

    # ── PartSupp ─────────────────────────────────────────────────────────
    print("Loading part-supplier relationships...")
    seen = set()
    for part_id in range(1, NUM_PARTS+1):
        for _ in range(4):
            sup_id = random.randint(1, NUM_SUPPLIERS)
            if (part_id, sup_id) not in seen:
                seen.add((part_id, sup_id))
                cur.execute(
                    "INSERT INTO partsupp VALUES (%s,%s,%s,%s,%s)",
                    (
                        part_id, sup_id,
                        random.randint(1, 9999),
                        round(random.uniform(1, 1000), 2),
                        fake.sentence()[:199]
                    )
                )
    conn.commit()

    # ── Orders + LineItems ────────────────────────────────────────────────
    print(f"Loading {NUM_ORDERS} orders with line items...")
    for i in range(1, NUM_ORDERS+1):
        order_date = random_date()
        cur.execute(
            "INSERT INTO orders VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                i,
                random.randint(1, NUM_CUSTOMERS),
                random.choice(["F","O","P"]),
                round(random.uniform(1000, 500000), 2),
                order_date,
                random.choice(PRIORITIES),
                f"Clerk#{random.randint(1,1000):09d}",
                0,
                fake.sentence()[:79]
            )
        )
        # Add 1-7 line items per order
        num_lines = random.randint(1, 7)
        used_parts = random.sample(range(1, NUM_PARTS+1), min(num_lines, NUM_PARTS))
        for line_num, part_id in enumerate(used_parts, 1):
            sup_id   = random.randint(1, NUM_SUPPLIERS)
            qty      = round(random.uniform(1, 50), 2)
            price    = round(random.uniform(100, 10000), 2)
            ship_d   = order_date + timedelta(days=random.randint(1, 60))
            commit_d = order_date + timedelta(days=random.randint(15, 30))
            recv_d   = ship_d   + timedelta(days=random.randint(1, 30))
            cur.execute(
                """INSERT INTO lineitem VALUES
                   (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    i, part_id, sup_id, line_num,
                    qty, price,
                    round(random.uniform(0, 0.10), 2),
                    round(random.uniform(0, 0.08), 2),
                    random.choice(["A","R","N"]),
                    random.choice(["F","O"]),
                    ship_d, commit_d, recv_d,
                    "DELIVER IN PERSON",
                    random.choice(SHIP_MODES),
                    fake.sentence()[:44]
                )
            )
        if i % 500 == 0:
            conn.commit()
            print(f"  {i}/{NUM_ORDERS} orders done...")

    conn.commit()

    # ── Add indexes for faster queries ────────────────────────────────────
    print("Creating indexes...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_custkey   ON orders(o_custkey);
        CREATE INDEX IF NOT EXISTS idx_orders_date      ON orders(o_orderdate);
        CREATE INDEX IF NOT EXISTS idx_lineitem_order   ON lineitem(l_orderkey);
        CREATE INDEX IF NOT EXISTS idx_lineitem_ship    ON lineitem(l_shipdate);
        CREATE INDEX IF NOT EXISTS idx_customer_segment ON customer(c_mktsegment);
        CREATE INDEX IF NOT EXISTS idx_supplier_nation  ON supplier(s_nationkey);
    """)
    conn.commit()

    # ── Verify row counts ─────────────────────────────────────────────────
    print("\n=== Data loaded successfully! ===")
    for table in ["region","nation","supplier","customer",
                  "part","partsupp","orders","lineitem"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:12s}: {count:>8,} rows")

    cur.close()
    conn.close()

if __name__ == "__main__":
    start = time.time()
    load_data()
    print(f"\nTotal time: {time.time()-start:.1f} seconds")
    print("Ready for Phase 2 data collection!")