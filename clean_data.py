"""
Week 1 – Data Acquisition, Cleaning, and Preprocessing
Dataset: Online Retail transaction data (UK-based e-commerce retailer)

Runs the full pipeline and saves:
  - cleaned dataset:      online_retail_cleaned.csv
  - summary stats (txt):  cleaning_summary.txt
  - plots (png):          fig_missing.png, fig_outliers_before_after.png,
                           fig_quantity_dist.png, fig_country.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
plt.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------------- load ----
df = pd.read_csv("/home/claude/week1/online_retail_raw.csv")
raw_shape = df.shape
log = []
log.append(f"Raw dataset shape: {raw_shape[0]} rows x {raw_shape[1]} columns")

# ---------------------------------------------------- initial exploration -
log.append("\nColumn dtypes (raw):\n" + str(df.dtypes))
log.append("\nMissing values per column (raw):\n" + str(df.isna().sum()))
log.append(f"\nExact duplicate rows (raw): {df.duplicated().sum()}")

missing_before = df.isna().sum()

fig, ax = plt.subplots(figsize=(6, 4))
missing_before[missing_before > 0].plot(kind="bar", ax=ax, color="#c0392b")
ax.set_title("Missing Values by Column (Before Cleaning)")
ax.set_ylabel("Count of missing values")
plt.tight_layout()
plt.savefig("/home/claude/week1/fig_missing.png")
plt.close()

# ------------------------------------------------------- 1. text cleanup --
for col in ["Description", "Country"]:
    df[col] = df[col].astype(str).str.strip()
    df.loc[df[col].isin(["nan", "NaN", ""]), col] = np.nan
df["Description"] = df["Description"].str.upper()
df["Country"] = df["Country"].str.title()
df["Country"] = df["Country"].replace({"Eire": "EIRE"})

# ------------------------------------------------ 2. parse InvoiceDate ----
def parse_mixed_date(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%m-%d-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

df["InvoiceDate"] = df["InvoiceDate"].apply(parse_mixed_date)
log.append(f"\nUnparseable dates after cleanup: {df['InvoiceDate'].isna().sum()}")

# ------------------------------------------- 3. cancelled-order handling --
df["IsCancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")
n_cancelled = df["IsCancelled"].sum()
log.append(f"\nCancelled-order line items flagged (InvoiceNo starts with 'C'): {n_cancelled}")

# --------------------------------------------------- 4. duplicate removal -
dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
log.append(f"\nExact duplicate rows removed: {dupes}")

# ------------------------------------------------ 5. missing Description --
# For rows with a missing description but a StockCode that appears
# elsewhere with a valid description, fill from the most common
# description for that StockCode (a real product lookup).
code_to_desc = (df.dropna(subset=["Description"])
                   .groupby("StockCode")["Description"]
                   .agg(lambda x: x.value_counts().idxmax()))
before_na = df["Description"].isna().sum()
df["Description"] = df.apply(
    lambda r: code_to_desc.get(r["StockCode"], r["Description"])
    if pd.isna(r["Description"]) else r["Description"], axis=1)
after_na = df["Description"].isna().sum()
log.append(f"\nMissing Description filled via StockCode lookup: {before_na - after_na} "
           f"(remaining unresolved: {after_na})")
df = df.dropna(subset=["Description"])  # drop any still unresolved

# --------------------------------------------------- 6. missing CustomerID
# CustomerID is missing for guest checkouts. We do not invent an identity;
# instead we keep the rows (they're still valid sales) but flag them,
# since dropping ~8% of transactions would bias revenue/EDA figures for
# later weeks. A placeholder "Guest" flag is added instead of imputing a
# numeric ID that would falsely imply a known customer.
df["IsGuestCheckout"] = df["CustomerID"].isna()
log.append(f"\nRows with missing CustomerID (flagged, not dropped): {df['IsGuestCheckout'].sum()}")

# ------------------------------------------------- 7. outlier treatment ---
q_before = df["Quantity"].copy()
p_before = df["UnitPrice"].copy()

# Zero/negative prices on non-cancelled rows are data-entry errors, not
# real transactions -> drop. Negative quantities are legitimate for
# cancelled orders (handled above via IsCancelled) so they are kept.
bad_price = (df["UnitPrice"] <= 0) & (~df["IsCancelled"])
log.append(f"\nRows with zero/negative UnitPrice on non-cancelled orders (dropped): {bad_price.sum()}")
df = df[~bad_price]

# Cap extreme Quantity and UnitPrice values using the IQR rule rather than
# deleting them outright, since large bulk orders are plausible for a
# wholesale-facing retailer; capping preserves the row while limiting
# distortion to later aggregate statistics.
def iqr_bounds(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

qty_lo, qty_hi = iqr_bounds(df.loc[~df["IsCancelled"], "Quantity"])
price_lo, price_hi = iqr_bounds(df["UnitPrice"])

n_qty_capped = ((df["Quantity"] > qty_hi) & (~df["IsCancelled"])).sum()
n_price_capped = (df["UnitPrice"] > price_hi).sum()

df.loc[(~df["IsCancelled"]) & (df["Quantity"] > qty_hi), "Quantity"] = int(qty_hi)
df.loc[df["UnitPrice"] > price_hi, "UnitPrice"] = round(price_hi, 2)

log.append(f"\nQuantity outliers capped at IQR upper bound ({qty_hi:.1f}): {n_qty_capped} rows")
log.append(f"UnitPrice outliers capped at IQR upper bound ({price_hi:.2f}): {n_price_capped} rows")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].boxplot([q_before.clip(upper=q_before.quantile(0.999)), df["Quantity"]],
                labels=["Before", "After"])
axes[0].set_title("Quantity: Before vs After Capping")
axes[1].boxplot([p_before.clip(upper=p_before.quantile(0.999)), df["UnitPrice"]],
                labels=["Before", "After"])
axes[1].set_title("UnitPrice: Before vs After Capping")
plt.tight_layout()
plt.savefig("/home/claude/week1/fig_outliers_before_after.png")
plt.close()

# ------------------------------------------------------- 8. derived field -
df["TotalPrice"] = (df["Quantity"] * df["UnitPrice"]).round(2)

# --------------------------------------------------------- 9. dtypes -----
df["StockCode"] = df["StockCode"].astype(str)
df["CustomerID"] = df["CustomerID"].astype("Int64")  # nullable integer

log.append("\nColumn dtypes (cleaned):\n" + str(df.dtypes))
log.append(f"\nFinal missing values per column:\n{df.isna().sum()}")
log.append(f"\nFinal cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns "
           f"(started at {raw_shape[0]} rows)")

# ------------------------------------------------------------- plots -----
fig, ax = plt.subplots(figsize=(6, 4))
df.loc[~df["IsCancelled"], "Quantity"].plot(kind="hist", bins=30, ax=ax, color="#2980b9")
ax.set_title("Distribution of Order Quantity (Cleaned, Non-Cancelled)")
ax.set_xlabel("Quantity")
plt.tight_layout()
plt.savefig("/home/claude/week1/fig_quantity_dist.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df["Country"].value_counts().head(10).plot(kind="bar", ax=ax, color="#27ae60")
ax.set_title("Top 10 Countries by Number of Transactions")
plt.tight_layout()
plt.savefig("/home/claude/week1/fig_country.png")
plt.close()

# -------------------------------------------------------------- save -----
df.to_csv("/home/claude/week1/online_retail_cleaned.csv", index=False)

with open("/home/claude/week1/cleaning_summary.txt", "w") as f:
    f.write("\n".join(log))

print("\n".join(log))
print("\nSaved cleaned dataset and figures.")
