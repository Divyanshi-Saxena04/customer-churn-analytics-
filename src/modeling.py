"""Week 3-4: predictive modelling, evaluation and feature importance."""
import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/customer_churn_clean.csv")
y = df["churn"]; X = df.drop(columns=["churn", "customer_id"])
num = X.select_dtypes("number").columns.tolist(); cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {"Logistic Regression": LogisticRegression(max_iter=1000),
          "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42),
          "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
rows, fitted, probs = [], {}, {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
    cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc").mean()
    rows.append(dict(model=name, accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                     recall=recall_score(yte, pred), f1=f1_score(yte, pred),
                     roc_auc=roc_auc_score(yte, p), cv_auc=cv))
    fitted[name], probs[name] = pipe, p
res = pd.DataFrame(rows).round(3); res.to_csv("outputs/model_comparison.csv", index=False); print(res)

# Fig 6: ROC curves
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for n, p in probs.items():
    f, t, _ = roc_curve(yte, p); ax.plot(f, t, label=f"{n} (AUC={roc_auc_score(yte, p):.3f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Figure 6: ROC Curves"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("images/fig6_roc.png", dpi=150); plt.close()

# best model by AUC
best = res.sort_values("roc_auc", ascending=False).iloc[0]["model"]; pipe = fitted[best]; p = probs[best]
# Fig 7: confusion matrix at tuned threshold (maximise F1)
ths = np.linspace(0.1, 0.7, 61); f1s = [f1_score(yte, (p >= t).astype(int)) for t in ths]
th = float(ths[int(np.argmax(f1s))])
cm = confusion_matrix(yte, (p >= th).astype(int))
fig, ax = plt.subplots(figsize=(4.8, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Figure 7: Confusion Matrix ({best}, t={th:.2f})", fontsize=9)
plt.tight_layout(); plt.savefig("images/fig7_confusion.png", dpi=150); plt.close()

# Fig 8: permutation importance
pi = permutation_importance(pipe, Xte, yte, n_repeats=10, random_state=42, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4.5)); imp.plot.barh(ax=ax, color="#8172B2")
ax.set_title("Figure 8: Permutation Feature Importance"); ax.set_xlabel("Drop in AUC when shuffled")
plt.tight_layout(); plt.savefig("images/fig8_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).round(4).to_csv("outputs/feature_importance.csv")

# Fig 9: lift / decile analysis for targeting
d = pd.DataFrame({"y": yte.values, "p": p}); d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = d.groupby("decile", observed=True)["y"].mean() / d.y.mean()
fig, ax = plt.subplots(figsize=(6, 4)); lift.plot.bar(ax=ax, color="#CCB974"); ax.axhline(1, color="k", ls="--")
ax.set_title("Figure 9: Lift by Risk Decile"); ax.set_ylabel("Lift vs. average"); ax.set_xlabel("Risk decile (1 = highest)")
plt.tight_layout(); plt.savefig("images/fig9_lift.png", dpi=150); plt.close()

top20 = d[d.decile.isin([1, 2])]
capture = top20.y.sum() / d.y.sum()
json.dump(dict(best_model=best, threshold=th, churn_rate=float(y.mean()), arpu=float(df.monthly_charges.mean()),
               top20_capture=float(capture), top_decile_lift=float(lift.iloc[0]), n_train=len(Xtr), n_test=len(Xte),
               tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
               top_features=imp.sort_values(ascending=False).head(5).index.tolist()),
          open("outputs/metrics.json", "w"), indent=2)
print(open("outputs/metrics.json").read())
