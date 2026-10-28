"""Generate a reproducible synthetic telecom customer-churn dataset.
Replace data/customer_churn.csv with your own internship data if you have it;
the rest of the pipeline only needs the same column names."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 5000

contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
internet = rng.choice(["Fiber optic", "DSL", "No internet"], N, p=[0.44, 0.36, 0.20])
tenure = np.where(contract == "Month-to-month", rng.integers(1, 40, N),
         np.where(contract == "One year", rng.integers(6, 60, N), rng.integers(12, 72, N)))
base = np.select([internet == "DSL", internet == "Fiber optic"], [45, 80], 20)
monthly = np.round(base + rng.normal(0, 6, N) + rng.integers(0, 3, N) * 5, 2).clip(18)
payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N,
                     p=[0.35, 0.18, 0.24, 0.23])
tech_support = np.where(internet == "No internet", "No internet", rng.choice(["Yes", "No"], N, p=[0.35, 0.65]))
paperless = rng.choice(["Yes", "No"], N, p=[0.6, 0.4])
senior = rng.choice([0, 1], N, p=[0.84, 0.16])
support_calls = rng.poisson(1.2 + 0.8 * (internet == "Fiber optic"), N)

logit = (-1.0
         + 1.3 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
         - 0.045 * tenure
         + 0.9 * (internet == "Fiber optic")
         + 0.012 * (monthly - 60)
         + 0.7 * (payment == "Electronic check")
         - 0.6 * (tech_support == "Yes")
         + 0.25 * senior + 0.2 * (paperless == "Yes")
         + 0.18 * support_calls)
churn = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(N)],
    "tenure_months": tenure, "contract": contract, "internet_service": internet,
    "monthly_charges": monthly, "total_charges": np.round(monthly * tenure, 2),
    "payment_method": payment, "tech_support": tech_support,
    "paperless_billing": paperless, "senior_citizen": senior,
    "support_calls": support_calls, "churn": churn})
# inject a little missingness to demonstrate cleaning
df.loc[rng.choice(N, 60, replace=False), "total_charges"] = np.nan
df.to_csv("data/customer_churn.csv", index=False)
print(df.shape, "churn rate:", round(df.churn.mean(), 3))
