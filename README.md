// Prédiction des prix des appartements - Régression linéaire

// Description
Ce projet est un TP réalisé dans le cadre de ma licence. Il vise à prédire les prix des appartements (vente et location) en Tunisie à partir de caractéristiques comme la surface, le nombre de chambres, l'étage, la présence d'ascenseur et la ville.

//  Structure du projet
- tp2_regression.py : Code principal avec implémentation complète
- prix_appartements.csv : Dataset des appartements
- requirements.txt : Bibliothèques nécessaires

//Features
- Surface (m²)
- Nombre de chambres
- Étage
- Ascenseur (Oui/Non)
- Ville (Tunis, Sousse, Sfax)
- Prix (cible)
- Catégorie (A Vendre / A Louer)

// Bibliothèques utilisées
- pandas : manipulation des données
- numpy : calculs numériques
- matplotlib : visualisations
- scikit-learn : modèles de régression
- joblib : sauvegarde des modèles

// Exécution
```bash
pip install -r requirements.txt
python tp2_regression.py