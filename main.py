import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Data Loading
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Model Configuration
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=141,
    learning_rate=0.05,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# 3. Cross-Validation
risultati = {}
print("Cross-Validation running...")

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results[name] = scores
    print(f"{name}: Media = {scores.mean():.4f} (+/- {scores.std():.4f})")

# 4. Boxplot Generation
plt.figure(figsize=(10, 6))
plt.boxplot(results.values(), labels=results.keys(), patch_artist=True)
plt.title('Final Comparison: Random Forest vs XGBoost')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save
plt.savefig('rf_vs_xgboost.png')

plt.show()

