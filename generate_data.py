"""
Generates a synthetic dataset that mirrors the structure and typical
data-quality problems of the UCI "Online Retail" dataset (invoice-level
e-commerce transactions for a UK-based online retailer).

NOTE: This sandbox has no internet access, so the real dataset could not
be downloaded. This script builds a same-shaped stand-in with deliberately
injected missing values, duplicates, outliers and formatting problems so
the cleaning pipeline below is fully runnable end-to-end. See the report
for a link to the real dataset if you want to swap it in later.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N = 5000

products = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("71270", "PINK CRYSTAL GUITAR PHONE CHARM", 1.25),
    ("22960", "JAM MAKING SET WITH JARS", 4.95),
    ("21755", "LOVE BUILDING BLOCK WORD", 5.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
]

countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain",
             "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia"]

start = datetime(2019, 1, 1)

rows = []
for i in range(N):
    code, desc, base_price = products[rng.integers(0, len(products))]
    invoice_no = 536000 + i // 3          # several line items share an invoice
    qty = int(rng.integers(1, 20))
    price = round(base_price * rng.uniform(0.9, 1.1), 2)
    date = start + timedelta(days=int(rng.integers(0, 365)),
                              minutes=int(rng.integers(0, 1440)))
    customer_id = float(rng.integers(12346, 18287))
    country = countries[rng.integers(0, len(countries))]
    rows.append([invoice_no, code, desc, qty, date, price, customer_id, country])

df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description",
                                  "Quantity", "InvoiceDate", "UnitPrice",
                                  "CustomerID", "Country"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)

# --- inject realistic data-quality problems -------------------------------

# 1. Missing CustomerID (guest checkouts) ~ 8%
mask = rng.random(N) < 0.08
df.loc[mask, "CustomerID"] = np.nan

# 2. Missing Description ~ 1%
mask = rng.random(N) < 0.01
df.loc[mask, "Description"] = np.nan

# 3. Cancelled orders: negative quantity, invoice prefixed with 'C'
cancel_idx = rng.choice(N, size=int(N * 0.03), replace=False)
df.loc[cancel_idx, "Quantity"] = -df.loc[cancel_idx, "Quantity"]
df.loc[cancel_idx, "InvoiceNo"] = "C" + df.loc[cancel_idx, "InvoiceNo"].astype(str)

# 4. Outlier quantities (data-entry errors, e.g. bulk-order typos)
outlier_idx = rng.choice(N, size=15, replace=False)
df.loc[outlier_idx, "Quantity"] = rng.integers(2000, 8000, size=15)

# 5. Outlier / erroneous unit prices (zero or absurdly high)
zero_price_idx = rng.choice(N, size=10, replace=False)
df.loc[zero_price_idx, "UnitPrice"] = 0.0
high_price_idx = rng.choice(N, size=8, replace=False)
df.loc[high_price_idx, "UnitPrice"] = rng.uniform(500, 2000, size=8).round(2)

# 6. Exact duplicate rows
dup_rows = df.sample(n=40, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# 7. Inconsistent text casing / stray whitespace in Description & Country
def messy_text(x):
    if pd.isna(x):
        return x
    r = rng.random()
    if r < 0.15:
        return x.lower()
    if r < 0.30:
        return "  " + x + "  "
    return x

df["Description"] = df["Description"].apply(messy_text)
df["Country"] = df["Country"].apply(messy_text)

# 8. InvoiceDate stored as mixed string formats (common real-world mess)
def format_date(d):
    r = rng.random()
    if r < 0.5:
        return d.strftime("%Y-%m-%d %H:%M:%S")
    elif r < 0.8:
        return d.strftime("%d/%m/%Y %H:%M")
    else:
        return d.strftime("%m-%d-%Y")

df["InvoiceDate"] = df["InvoiceDate"].apply(format_date)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("/home/claude/week1/online_retail_raw.csv", index=False)
print("Saved", df.shape, "rows/cols")
print(df.head())
