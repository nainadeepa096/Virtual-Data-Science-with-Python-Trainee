"""
Week 2 - Exploratory Data Analysis and Visualization
Input: online_retail_cleaned.csv (output of the Week 1 cleaning pipeline)

Produces figures (png) and a text summary (eda_summary.txt) used in the
Week 2 report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
OUT = "/home/claude/week2/"

df = pd.read_csv(OUT + "online_retail_cleaned.csv", parse_dates=["InvoiceDate"])

log = []
log.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Work with completed (non-cancelled) sales for revenue/demand analysis,
# consistent with the note in the Week 1 report.
sales = df[~df["IsCancelled"]].copy()
log.append(f"Completed-sale rows used for revenue/demand analysis: {len(sales)} "
            f"({df['IsCancelled'].sum()} cancelled rows excluded)")

# ---------------------------------------------------------- 1. summary ----
desc = sales[["Quantity", "UnitPrice", "TotalPrice"]].describe().round(2)
log.append("\nSummary statistics (completed sales):\n" + str(desc))

guest_share = df["IsGuestCheckout"].mean()
log.append(f"\nShare of guest-checkout transactions: {guest_share:.1%}")

# ------------------------------------------------- 2. revenue over time ---
sales["Month"] = sales["InvoiceDate"].dt.to_period("M").astype(str)
monthly = sales.groupby("Month")["TotalPrice"].sum().sort_index()
log.append("\nMonthly revenue:\n" + str(monthly.round(2)))

fig, ax = plt.subplots(figsize=(9, 4.2))
monthly.plot(kind="line", marker="o", ax=ax, color="#2980b9")
ax.set_title("Monthly Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (\u00a3)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT + "fig_monthly_revenue.png")
plt.close()

# --------------------------------------------- 3. top products by revenue -
top_products = (sales.groupby("Description")["TotalPrice"]
                 .sum().sort_values(ascending=False).head(10))
log.append("\nTop 10 products by revenue:\n" + str(top_products.round(2)))

fig, ax = plt.subplots(figsize=(8, 5))
top_products.sort_values().plot(kind="barh", ax=ax, color="#8e44ad")
ax.set_title("Top 10 Products by Revenue")
ax.set_xlabel("Revenue (\u00a3)")
plt.tight_layout()
plt.savefig(OUT + "fig_top_products.png")
plt.close()

# ------------------------------------------------- 4. revenue by country --
top_countries = (sales.groupby("Country")["TotalPrice"]
                  .sum().sort_values(ascending=False).head(10))
log.append("\nTop 10 countries by revenue:\n" + str(top_countries.round(2)))

fig, ax = plt.subplots(figsize=(8, 4.5))
top_countries.plot(kind="bar", ax=ax, color="#16a085")
ax.set_title("Top 10 Countries by Revenue")
ax.set_ylabel("Revenue (\u00a3)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT + "fig_country_revenue.png")
plt.close()

# ---------------------------------------- 5. distribution & relationship --
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(sales["UnitPrice"], bins=30, ax=axes[0], color="#e67e22")
axes[0].set_title("Distribution of Unit Price")
sns.histplot(sales["TotalPrice"], bins=30, ax=axes[1], color="#2c3e50")
axes[1].set_title("Distribution of Order Line Value (TotalPrice)")
plt.tight_layout()
plt.savefig(OUT + "fig_price_distributions.png")
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.scatterplot(data=sales.sample(min(1500, len(sales)), random_state=1),
                 x="Quantity", y="UnitPrice", alpha=0.4, ax=ax, color="#c0392b")
ax.set_title("Quantity vs. Unit Price")
plt.tight_layout()
plt.savefig(OUT + "fig_qty_vs_price.png")
plt.close()

corr = sales[["Quantity", "UnitPrice", "TotalPrice"]].corr().round(3)
log.append("\nCorrelation matrix (Quantity, UnitPrice, TotalPrice):\n" + str(corr))

fig, ax = plt.subplots(figsize=(5, 4.2))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT + "fig_correlation_heatmap.png")
plt.close()

# --------------------------------------------- 6. day-of-week seasonality -
sales["DayOfWeek"] = sales["InvoiceDate"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_rev = sales.groupby("DayOfWeek")["TotalPrice"].sum().reindex(dow_order)
log.append("\nRevenue by day of week:\n" + str(dow_rev.round(2)))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
dow_rev.plot(kind="bar", ax=ax, color="#2980b9")
ax.set_title("Revenue by Day of Week")
ax.set_ylabel("Revenue (\u00a3)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT + "fig_day_of_week.png")
plt.close()

# ------------------------------------------- 7. guest vs. registered -----
guest_stats = df.groupby("IsGuestCheckout")["TotalPrice"].agg(["mean", "count"]).round(2)
log.append("\nAverage order-line value, guest vs. registered:\n" + str(guest_stats))

with open(OUT + "eda_summary.txt", "w") as f:
    f.write("\n".join(log))

print("\n".join(log))
print("\nSaved EDA figures and summary.")
