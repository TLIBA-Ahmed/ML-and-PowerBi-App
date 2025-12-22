# 🤖 Django ML Analysis Hub

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Django Version](https://img.shields.io/badge/django-5.0-green.svg)](https://www.djangoproject.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)

Cette application Django est une plateforme intégrée permettant d'exécuter, de visualiser et d'interagir avec trois types d'analyses de Machine Learning issues de Notebooks Jupyter.

---

## 🚀 Fonctionnalités Principales

### 1. Analyse de Clustering (`Clustering.ipynb`)
* **Méthodologie** : Utilisation de K-means avec réduction de dimensionnalité **PCA**.
* **Visualisations** : Graphiques de la méthode du coude (Elbow Method) et profils de clusters.
* **Prédiction** : Interface pour assigner un cluster à un nouveau job IA.

### 2. Prédiction de Salaire (`DS1.ipynb`)
Comparaison en temps réel de 5 modèles de régression :
* Régression Linéaire & Polynomiale
* Arbre de Décision
* **Random Forest** & **Gradient Boosting**
* **Inclus** : Matrice de corrélation et outils de prédiction personnalisée.

### 3. Classification des Plateformes (`Classification.ipynb`)
Détermination de la meilleure plateforme de recrutement (Accuracy max: **83.7%** via **XGBoost**).
* Modèles supportés : XGBoost, Random Forest, KNN, SVM, Decision Tree.
* Sortie : Probabilités par plateforme et matrice de confusion.

### 4. Historique & Persistance
* Sauvegarde automatique de toutes les prédictions en base de données.
* Tableau de bord de consultation des analyses précédentes.

---

## 🛠️ Installation et Configuration

### Prérequis
* Python 3.10+
* `pip` (gestionnaire de paquets)

### Étapes d'installation

1. **Cloner le projet**
   ```bash
   git clone [https://github.com/TLIBA-Ahmed/ML-and-PowerBi-App.git](https://github.com/TLIBA-Ahmed/ML-and-PowerBi-App.git)
   cd nom-du-repo

