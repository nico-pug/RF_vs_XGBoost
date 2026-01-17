import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Caricamento Dati
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Configurazione Modelli
# Random Forest: Robusta e semplice
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost: Configuriamo per evitare warning e usare la metrica logloss
xgb_model = xgb.XGBClassifier(
    n_estimators=141,        # Usiamo il numero ottimale che hai trovato!
    learning_rate=0.05,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

modelli = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# 3. Cross-Validation (Testiamo 10 volte su dati diversi)
risultati = {}
print("Esecuzione Cross-Validation in corso...")

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for nome, modello in modelli.items():
    scores = cross_val_score(modello, X, y, cv=kfold, scoring='accuracy')
    risultati[nome] = scores
    print(f"{nome}: Media = {scores.mean():.4f} (+/- {scores.std():.4f})")

# 4. Generazione Grafico (Boxplot)
plt.figure(figsize=(10, 6))
plt.boxplot(risultati.values(), labels=risultati.keys(), patch_artist=True)
plt.title('Confronto Finale: Random Forest vs XGBoost')
plt.ylabel('Accuratezza')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Salva il grafico per il README
plt.savefig('confronto_rf_xgboost.png')
plt.show()