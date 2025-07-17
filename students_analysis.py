# students_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Chargement et nettoyage
df = pd.read_csv(r'C:\Users\Waad RTIBI\students_analysis\StudentsPerformance.csv')
print("🔹 Aperçu des données :")
print(df.head())

print("\n🔍 Valeurs manquantes :")
print(df.isnull().sum())

df.dropna(inplace=True)

df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# 2. Visualisation

sns.set(style="whitegrid")

# Histogramme
plt.figure(figsize=(8, 5))
sns.histplot(df["average_score"], kde=True, bins=20, color="skyblue")
plt.title("Distribution du score moyen")
plt.xlabel("Score moyen")
plt.ylabel("Nombre d'élèves")
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x="test preparation course", y="average_score", data=df, palette="Set2")
plt.title("Score moyen selon la préparation au test")
plt.xlabel("Préparation au test")
plt.ylabel("Score moyen")
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="math score", y="reading score", hue="gender", palette="Set1")
plt.title("Math vs Lecture par genre")
plt.xlabel("Note en math")
plt.ylabel("Note en lecture")
plt.tight_layout()
plt.show()

# 3. Machine Learning : prédiction

# Variable cible : réussite
df["passed"] = (df["average_score"] >= 60).astype(int)

# Variables explicatives
features = ["gender", "test preparation course", "math score", "reading score", "writing score"]
X = df[features].copy()
y = df["passed"]

# Encodage des variables catégorielles
le_gender = LabelEncoder()
le_prep = LabelEncoder()

X["gender"] = le_gender.fit_transform(X["gender"])
X["test preparation course"] = le_prep.fit_transform(X["test preparation course"])

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Précision du modèle Random Forest : {accuracy:.2%}")
