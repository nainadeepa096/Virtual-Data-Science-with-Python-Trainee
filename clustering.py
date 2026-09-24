"""
Week 3 - Unsupervised Learning and Clustering Analysis
Input: online_retail_cleaned.csv (Week 1 output)

Builds a per-customer feature table (RFM-style: Recency, Frequency,
Monetary + average order value), then applies K-Means (with an elbow +
silhouette check to choose k) and Agglomerative (hierarchical) clustering
for comparison, visualizing clusters via PCA.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
OUT = "/home/claude/week3/"

df = pd.read_csv(OUT + "online_retail_cleaned.csv", parse_dates=["InvoiceDate"])
log = []

# Only registered customers (non-guest, non-cancelled) can be attributed
# to a CustomerID, so clustering is done at the customer level using that
# subset - consistent with the guest-checkout flag introduced in Week 1.
sales = df[(~df["IsCancelled"]) & (~df["IsGuestCheckout"])].copy()
log.append(f"Rows used for customer-level feature engineering: {len(sales)} "
            f"(excludes cancelled orders and guest checkouts)")

# --------------------------------------------------- 1. RFM feature table -
snapshot_date = sales["InvoiceDate"].max() + pd.Timedelta(days=1)

cust = sales.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    AvgOrderValue=("TotalPrice", "mean"),
    TotalItems=("Quantity", "sum"),
).reset_index()

log.append(f"\nCustomer-level table: {cust.shape[0]} customers x {cust.shape[1]-1} features")
log.append("\nFeature summary:\n" + str(cust.describe().round(2)))

features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalItems"]
X = cust[features].values
X_scaled = StandardScaler().fit_transform(X)

# ------------------------------------------------- 2. choose k (elbow) ----
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

log.append("\nInertia and silhouette score by k:")
for k, i, s in zip(K_range, inertias, sil_scores):
    log.append(f"  k={k}: inertia={i:.1f}, silhouette={s:.3f}")

best_by_silhouette = list(K_range)[int(np.argmax(sil_scores))]
log.append(f"\nk with highest silhouette score: {best_by_silhouette} (score={max(sil_scores):.3f})")

# k=2 scores highest but only separates "light" vs "heavier" buyers, which
# is too coarse to be actionable for marketing/segmentation purposes.
# k=4 is chosen instead: it is the next local silhouette peak, the elbow
# plot also flattens noticeably by k=4, and it yields customer segments
# with distinct, business-interpretable profiles (see Section 4).
best_k = 4
log.append(f"Chosen k for the final model: {best_k} (see report Section 3 for rationale)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(list(K_range), inertias, marker="o", color="#2980b9")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), sil_scores, marker="o", color="#c0392b")
axes[1].axvline(best_k, color="gray", linestyle="--", alpha=0.6)
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig(OUT + "fig_elbow_silhouette.png")
plt.close()

# ------------------------------------------------------- 3. K-Means -------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cust["KMeansCluster"] = kmeans.fit_predict(X_scaled)
km_sil = silhouette_score(X_scaled, cust["KMeansCluster"])
log.append(f"\nFinal K-Means (k={best_k}) silhouette score: {km_sil:.3f}")

cluster_profile = cust.groupby("KMeansCluster")[features].mean().round(2)
cluster_profile["Count"] = cust["KMeansCluster"].value_counts().sort_index()
log.append("\nK-Means cluster profile (mean feature values):\n" + str(cluster_profile))

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
cust["PCA1"], cust["PCA2"] = coords[:, 0], coords[:, 1]
log.append(f"\nPCA explained variance ratio (2 components): {pca.explained_variance_ratio_.round(3)}")

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=cust, x="PCA1", y="PCA2", hue="KMeansCluster",
                 palette="tab10", ax=ax, s=45, alpha=0.8)
ax.set_title(f"Customer Segments (K-Means, k={best_k}) - PCA Projection")
plt.tight_layout()
plt.savefig(OUT + "fig_kmeans_pca.png")
plt.close()

# ---------------------------------------------- 4. Hierarchical clustering
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
cust["HierCluster"] = agg.fit_predict(X_scaled)
hier_sil = silhouette_score(X_scaled, cust["HierCluster"])
log.append(f"\nAgglomerative (Ward linkage, k={best_k}) silhouette score: {hier_sil:.3f}")

# Dendrogram on a sample (full dendrograms are unreadable beyond ~50 leaves)
sample_idx = np.random.RandomState(1).choice(len(X_scaled), size=min(40, len(X_scaled)), replace=False)
Z = linkage(X_scaled[sample_idx], method="ward")
fig, ax = plt.subplots(figsize=(10, 4.5))
dendrogram(Z, ax=ax)
ax.set_title("Hierarchical Clustering Dendrogram (40-customer sample, Ward linkage)")
ax.set_xlabel("Customer index"); ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig(OUT + "fig_dendrogram.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=cust, x="PCA1", y="PCA2", hue="HierCluster",
                 palette="tab10", ax=ax, s=45, alpha=0.8)
ax.set_title(f"Customer Segments (Hierarchical, k={best_k}) - PCA Projection")
plt.tight_layout()
plt.savefig(OUT + "fig_hier_pca.png")
plt.close()

agreement = (cust["KMeansCluster"].astype(str) + cust["HierCluster"].astype(str))
log.append(f"\nK-Means vs. Hierarchical: {cust.groupby(['KMeansCluster','HierCluster']).size().shape[0]} "
            f"distinct (KMeans,Hier) label pairs across {best_k} clusters each")

# --------------------------------------------- 5. feature importance plot -
fig, ax = plt.subplots(figsize=(8, 4.5))
cluster_profile[features].T.plot(kind="bar", ax=ax)
ax.set_title("Mean Feature Values by K-Means Cluster")
ax.set_ylabel("Mean value (original units)")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(OUT + "fig_cluster_features.png")
plt.close()

cust.to_csv(OUT + "customer_segments.csv", index=False)
with open(OUT + "clustering_summary.txt", "w") as f:
    f.write("\n".join(log))

print("\n".join(log))
print("\nSaved clustering outputs.")
