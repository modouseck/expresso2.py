# =======================================
# 1. Importation des bibliothèques
# =======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =======================================
# 2. Importation du jeu de données
# =======================================
# Assurez-vous que le chemin du fichier est correct
df = pd.read_csv("/Users/mouhamedseck/Downloads/expresso.py/Expresso_churn_dataset.csv")                                                  
# Aperçu rapide
print(df.head())
print(df.info())
print(df.describe())

# =======================================
# 3. Rapport de profilage Pandas
# =======================================
profile = ProfileReport(df, title="Rapport Expresso Churn", explorative=True)
profile.to_file("rapport_churn.html")  # Ce fichier sera consultable dans ton navigateur

# =======================================
# 4. Nettoyage des données
# =======================================
# Valeurs manquantes
print(df.isnull().sum())
df.fillna(df.median(numeric_only=True), inplace=True)  # Remplacer valeurs manquantes numériques par la médiane
df.fillna(df.mode().iloc[0], inplace=True)  # Remplacer valeurs manquantes catégorielles par la valeur la plus fréquente

# Doublons
df.drop_duplicates(inplace=True)

# =======================================
# 5. Encodage des variables catégorielles
# =======================================
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

# =======================================
# 6. Séparation X et y
# =======================================
X = df.drop("CHURN", axis=1)  # "CHURN" = colonne cible
y = df["CHURN"]

# =======================================
# 7. Train / Test split
# =======================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================================
# 8. Modélisation
# =======================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =======================================
# 9. Évaluation
# =======================================
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()

# =======================================
# 10. Sauvegarde du modèle
# =======================================
import joblib
joblib.dump(model, "modele_churn.pkl")

import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("modele_churn.pkl")

st.title("Prédiction de Churn - Expresso")

# Champs de saisie pour chaque variable
input_data = {}
input_data["MONTANT"] = st.number_input("Montant", min_value=0.0)
input_data["FREQUENCE_RECH"] = st.number_input("Fréquence Recharges", min_value=0)
input_data["REVENUE"] = st.number_input("Revenu", min_value=0.0)
input_data["ARPU_SEGMENT"] = st.number_input("ARPU Segment", min_value=0.0)
# Ajoute tous les champs nécessaires ici…

# Bouton de prédiction
if st.button("Prédire"):
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)
    if prediction[0] == 1:
        st.error("⚠️ Le client risque de se désabonner")
    else:
        st.success("✅ Le client est fidèle")
st.write("Résumé des données :")
st.write(df.describe())   # statistiques
st.write(df.info())       # infos sur colonnes

# =======================================
# 4 bis. Visualisation des données avec Seaborn
# =======================================
#import matplotlib.pyplot as plt
import seaborn as sns

# Histogramme du churn
plt.figure(figsize=(6,4))
sns.countplot(x="CHURN", data=df, palette="Set2")
plt.title("Répartition du Churn (0 = Fidèle, 1 = Désabonné)")
plt.show()

# Boxplot : revenu vs churn
plt.figure(figsize=(6,4))
sns.boxplot(x="CHURN", y="REVENUE", data=df, palette="coolwarm")
plt.title("Revenu selon le Churn")
plt.show()

# Heatmap des corrélations
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Matrice de corrélations")
plt.show()

# Distribution du montant par churn
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="MONTANT", hue="CHURN", kde=True, palette="Set1")
plt.title("Montant des recharges selon le churn")
plt.show()

# Pairplot rapide pour quelques variables
colonnes_cles = ["MONTANT", "REVENUE", "FREQUENCE_RECH", "CHURN"]
sns.pairplot(df[colonnes_cles], hue="CHURN", palette="husl")
plt.show()

