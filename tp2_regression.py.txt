
TP 2 : Prédiction des prix des appartements avec la régression
Auteur: Rayen Ellouze
Description: Régression linéaire pour prédire les prix des appartements


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time


// 1. CHARGEMENT DES DONNÉES

print("=" * 50)
print("CHARGEMENT DES DONNÉES")
print("=" * 50)

debut_chargement = time.time()
df = pd.read_csv('prix_appartements.csv.txt')
temps_chargement = time.time() - debut_chargement
print(f"Temps de chargement : {temps_chargement:.2f} secondes")
print(f"Shape du dataset : {df.shape}")
print(f"Colonnes : {df.columns.tolist()}")
print("\nPremières lignes :")
print(df.head())


// 2. EXPLORATION DES DONNÉES

print("\n" + "=" * 50)
print("EXPLORATION DES DONNÉES")
print("=" * 50)

print("\nStatistiques descriptives :")
print(df.describe())

print("\nTypes de données :")
print(df.dtypes)

print("\nValeurs manquantes :")
print(df.isnull().sum())


// 3. PRÉTRAITEMENT

print("\n" + "=" * 50)
print("PRÉTRAITEMENT")
print("=" * 50)

// Suppression des valeurs aberrantes (prix > 3 écarts-types)
print("\nSuppression des outliers...")
df_clean = df[np.abs(df['Prix'] - df['Prix'].mean()) <= (3 * df['Prix'].std())]
print(f"Shape avant nettoyage : {df.shape}")
print(f"Shape après nettoyage : {df_clean.shape}")

// Séparation vente / location
print("\nSéparation vente / location...")
df_vente = df_clean[df_clean['Categorie'] == 'A Vendre']
df_location = df_clean[df_clean['Categorie'] == 'A Louer']

print(f"Nombre d'appartements à vendre : {len(df_vente)}")
print(f"Nombre d'appartements à louer : {len(df_location)}")

// Encodage des variables catégorielles
print("\nEncodage des variables catégorielles...")
df_vente_encoded = pd.get_dummies(df_vente, columns=['Ascenseur', 'Ville'], drop_first=True)
df_location_encoded = pd.get_dummies(df_location, columns=['Ascenseur', 'Ville'], drop_first=True)


//4. MODÈLE POUR LA VENTE

print("\n" + "=" * 50)
print("MODÈLE RÉGRESSION LINÉAIRE - VENTE")
print("=" * 50)

// Préparation des features (X) et target (y)
features_vente = ['Surface', 'Chambres', 'Etage']
// Ajouter les colonnes encodées si elles existent
for col in df_vente_encoded.columns:
    if col.startswith('Ascenseur_') or col.startswith('Ville_'):
        features_vente.append(col)

X_vente = df_vente_encoded[features_vente]
y_vente = df_vente_encoded['Prix']

//Split train/test
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vente, y_vente, test_size=0.2, random_state=42
)

print(f"Train set : {X_train_v.shape}")
print(f"Test set : {X_test_v.shape}")

// Entraînement
debut_train = time.time()
model_vente = LinearRegression()
model_vente.fit(X_train_v, y_train_v)
temps_train = time.time() - debut_train
print(f"Temps d'entraînement : {temps_train:.3f} secondes")

// Prédictions
y_pred_v = model_vente.predict(X_test_v)

//Évaluation
mse_v = mean_squared_error(y_test_v, y_pred_v)
r2_v = r2_score(y_test_v, y_pred_v)

print(f"\nRésultats - VENTE :")
print(f"MSE : {mse_v:.2f}")
print(f"R² : {r2_v:.3f}")


// 5. MODÈLE POUR LA LOCATION

print("\n" + "=" * 50)
print("MODÈLE RÉGRESSION LINÉAIRE - LOCATION")
print("=" * 50)

// Préparation des features
features_location = ['Surface', 'Chambres', 'Etage']
for col in df_location_encoded.columns:
    if col.startswith('Ascenseur_') or col.startswith('Ville_'):
        features_location.append(col)

X_location = df_location_encoded[features_location]
y_location = df_location_encoded['Prix']

// Split train/test
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_location, y_location, test_size=0.2, random_state=42
)

print(f"Train set : {X_train_l.shape}")
print(f"Test set : {X_test_l.shape}")

// Entraînement
model_location = LinearRegression()
model_location.fit(X_train_l, y_train_l)

// Prédictions
y_pred_l = model_location.predict(X_test_l)

// Évaluation
mse_l = mean_squared_error(y_test_l, y_pred_l)
r2_l = r2_score(y_test_l, y_pred_l)

print(f"\nRésultats - LOCATION :")
print(f"MSE : {mse_l:.2f}")
print(f"R² : {r2_l:.3f}")


// 6. ANALYSE DES COEFFICIENTS

print("\n" + "=" * 50)
print("ANALYSE DES COEFFICIENTS")
print("=" * 50)

print("\nCoefficients - VENTE :")
coef_vente = pd.DataFrame({
    'Feature': features_vente,
    'Coefficient': model_vente.coef_
})
print(coef_vente.sort_values('Coefficient', ascending=False))

print("\nCoefficients - LOCATION :")
coef_location = pd.DataFrame({
    'Feature': features_location,
    'Coefficient': model_location.coef_
})
print(coef_location.sort_values('Coefficient', ascending=False))


// 7. SAUVEGARDE DES MODÈLES

print("\n" + "=" * 50)
print("SAUVEGARDE DES MODÈLES")
print("=" * 50)

import joblib
joblib.dump(model_vente, 'model_vente.pkl')
joblib.dump(model_location, 'model_location.pkl')
print("Modèles sauvegardés : model_vente.pkl, model_location.pkl")

print("\n" + "=" * 50)
print("FIN DU TRAITEMENT")
print("=" * 50)