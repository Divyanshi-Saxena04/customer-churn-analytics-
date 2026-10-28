"""Week 1-2: data cleaning and exploratory data analysis. Saves figures to images/."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
df = pd.read_csv("data/customer_churn.csv")
print(df.info()); print(df.isna().sum()[lambda s: s > 0])

# --- cleaning: impute missing total_charges as monthly * tenure
df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])
df.to_csv("data/customer_churn_clean.csv", index=False)

# Fig 1: churn distribution
fig, ax = plt.subplots(figsize=(5, 4))
df["churn"].map({0: "Retained", 1: "Churned"}).value_counts().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_title("Figure 1: Customer Churn Distribution"); ax.set_ylabel("Customers"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("images/fig1_churn_distribution.png", dpi=150); plt.close()

# Fig 2: churn rate by contract
fig, ax = plt.subplots(figsize=(6, 4))
(df.groupby("contract")["churn"].mean() * 100).sort_values().plot.barh(ax=ax, color="#55A868")
ax.set_title("Figure 2: Churn Rate by Contract Type"); ax.set_xlabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("images/fig2_churn_by_contract.png", dpi=150); plt.close()

# Fig 3: tenure distribution by churn
fig, ax = plt.subplots(figsize=(6.5, 4))
sns.histplot(data=df, x="tenure_months", hue="churn", bins=30, multiple="layer", ax=ax)
ax.set_title("Figure 3: Tenure Distribution by Churn Status")
plt.tight_layout(); plt.savefig("images/fig3_tenure_hist.png", dpi=150); plt.close()

# Fig 4: churn by internet service and tech support
fig, ax = plt.subplots(figsize=(6.5, 4))
t = df.groupby(["internet_service", "tech_support"])["churn"].mean().mul(100).reset_index()
sns.barplot(data=t, x="internet_service", y="churn", hue="tech_support", ax=ax)
ax.set_title("Figure 4: Churn Rate by Internet Service & Tech Support"); ax.set_ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("images/fig4_internet_support.png", dpi=150); plt.close()

# Fig 5: correlation heatmap
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(df[["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen", "churn"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Figure 5: Correlation Matrix")
plt.tight_layout(); plt.savefig("images/fig5_correlation.png", dpi=150); plt.close()

# summary tables used in the report
df.groupby("contract")["churn"].agg(["count", "mean"]).to_csv("outputs/churn_by_contract.csv")
df.groupby("payment_method")["churn"].agg(["count", "mean"]).to_csv("outputs/churn_by_payment.csv")
df.groupby("internet_service")["churn"].agg(["count", "mean"]).to_csv("outputs/churn_by_internet.csv")
print("overall churn:", df.churn.mean())
