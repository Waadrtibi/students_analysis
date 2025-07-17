import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np

st.set_page_config(page_title="Analyse & Prédiction Étudiants", layout="wide")
st.title(" Analyse & Prédiction des Résultats des Étudiants")

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Waad RTIBI/students_analysis/StudentsPerformance.csv")

    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["passed"] = df["average_score"].apply(lambda x: 1 if x >= 60 else 0)
    return df

df = load_data()

df_encoded = pd.get_dummies(df, columns=["gender", "test preparation course"], drop_first=True)
features = ["math score", "reading score", "writing score", "gender_male", "test preparation course_none"]
X = df_encoded[features]
y = df_encoded["passed"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

st.sidebar.title(" Choix du Modèle")
model_name = st.sidebar.radio("Sélectionnez un modèle :", list(models.keys()))

# Entraîner tous les modèles pour comparaison
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "fpr": fpr,
        "tpr": tpr,
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# Afficher résultats du modèle choisi
st.header(f" Résultats pour : {model_name}")
st.write(f"**Accuracy** : {results[model_name]['accuracy']:.2f}")

# Matrice de confusion
fig_cm, ax_cm = plt.subplots()
sns.heatmap(results[model_name]["conf_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Prédit")
ax_cm.set_ylabel("Réel")
ax_cm.set_title("Matrice de Confusion")
st.pyplot(fig_cm)

# ROC Curve
fig_roc, ax_roc = plt.subplots()
for name, res in results.items():
    ax_roc.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {auc(res['fpr'], res['tpr']):.2f})")
ax_roc.plot([0, 1], [0, 1], 'k--', label="Random")
ax_roc.set_xlabel("Faux positif")
ax_roc.set_ylabel("Vrai positif")
ax_roc.set_title("Courbe ROC des Modèles")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Comparaison tableau
st.subheader(" Comparaison des Performances")
compare_df = pd.DataFrame({
    "Modèle": list(results.keys()),
    "Accuracy": [results[k]["accuracy"] for k in results]
}).sort_values("Accuracy", ascending=False)

st.dataframe(compare_df)

