# COURS DE SCIENCE DES DONNÉES

## École Nationale de Commerce et de Gestion (ENCG) \- 4ème Année

---

# PARTIE 2 : APPRENTISSAGE AUTOMATIQUE ET APPLICATIONS AVANCÉES

---

## MODULE 3 : APPRENTISSAGE AUTOMATIQUE (MACHINE LEARNING)

### 3.5.2 Clustering Hiérarchique

Le clustering hiérarchique crée une hiérarchie de clusters représentée par un dendrogramme.

**Avantages :**

- Pas besoin de spécifier k à l'avance  
- Visualisation intuitive  
- Plusieurs méthodes de liaison (linkage)

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering

print("=" \* 80\)

print("CLUSTERING HIÉRARCHIQUE")

print("=" \* 80\)

\# Utiliser un échantillon pour la visualisation

sample\_size \= 50

sample\_indices \= np.random.choice(len(X\_scaled), sample\_size, replace=False)

X\_sample \= X\_scaled\[sample\_indices\]

\# Calculer la matrice de liaison

Z \= linkage(X\_sample, method='ward')

\# Visualiser le dendrogramme

plt.figure(figsize=(14, 6))

dendrogram(Z, labels=sample\_indices)

plt.xlabel('Index des observations')

plt.ylabel('Distance')

plt.title('Dendrogramme \- Clustering Hiérarchique (Ward)')

plt.axhline(y=10, color='r', linestyle='--', label='Seuil de coupure')

plt.legend()

plt.grid(True, alpha=0.3, axis='y')

plt.show()

\# Appliquer le clustering hiérarchique

hierarchical \= AgglomerativeClustering(n\_clusters=3, linkage='ward')

hier\_clusters \= hierarchical.fit\_predict(X\_scaled)

df\_iris\['Hier\_Cluster'\] \= hier\_clusters

print("\\nComparaison K-Means vs Hiérarchique:")

comparison \= pd.crosstab(df\_iris\['Cluster'\], df\_iris\['Hier\_Cluster'\], 

                         rownames=\['K-Means'\], colnames=\['Hiérarchique'\])

print(comparison)

### 3.5.3 Réduction de Dimensionnalité : Analyse en Composantes Principales (ACP/PCA)

L'ACP transforme les variables corrélées en composantes principales non corrélées.

**Objectifs :**

- Réduire la dimensionnalité  
- Visualiser des données complexes  
- Éliminer le bruit  
- Accélérer les algorithmes

**Formule mathématique :**

Les composantes principales sont les vecteurs propres de la matrice de covariance des données standardisées.

from sklearn.decomposition import PCA

print("=" \* 80\)

print("ANALYSE EN COMPOSANTES PRINCIPALES (PCA)")

print("=" \* 80\)

\# Appliquer PCA

pca \= PCA()

X\_pca \= pca.fit\_transform(X\_scaled)

\# Variance expliquée

variance\_expliquee \= pca.explained\_variance\_ratio\_

variance\_cumulee \= np.cumsum(variance\_expliquee)

print("Variance expliquée par composante:")

for i, var in enumerate(variance\_expliquee):

    print(f"PC{i+1}: {var:.4f} ({var\*100:.2f}%)")

print(f"\\nVariance cumulée avec 2 composantes: {variance\_cumulee\[1\]:.4f} ({variance\_cumulee\[1\]\*100:.2f}%)")

\# Visualisation

fig, axes \= plt.subplots(1, 3, figsize=(18, 5))

\# 1\. Variance expliquée

axes\[0\].bar(range(1, len(variance\_expliquee)+1), variance\_expliquee, alpha=0.7)

axes\[0\].plot(range(1, len(variance\_cumulee)+1), variance\_cumulee, 

             'ro-', linewidth=2, markersize=8, label='Cumulée')

axes\[0\].set\_xlabel('Composante Principale')

axes\[0\].set\_ylabel('Variance Expliquée')

axes\[0\].set\_title('Variance Expliquée par Composante')

axes\[0\].legend()

axes\[0\].grid(True, alpha=0.3)

\# 2\. Projection sur PC1 et PC2 (par espèce réelle)

species\_map \= {species: i for i, species in enumerate(df\_iris\['species'\].unique())}

colors \= df\_iris\['species'\].map(species\_map)

scatter \= axes\[1\].scatter(X\_pca\[:, 0\], X\_pca\[:, 1\], c=colors, 

                          cmap='viridis', alpha=0.6, s=50)

axes\[1\].set\_xlabel(f'PC1 ({variance\_expliquee\[0\]\*100:.1f}%)')

axes\[1\].set\_ylabel(f'PC2 ({variance\_expliquee\[1\]\*100:.1f}%)')

axes\[1\].set\_title('Projection PCA \- Espèces Réelles')

axes\[1\].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes\[1\])

\# 3\. Projection sur PC1 et PC2 (par cluster)

scatter2 \= axes\[2\].scatter(X\_pca\[:, 0\], X\_pca\[:, 1\], c=clusters, 

                           cmap='plasma', alpha=0.6, s=50)

axes\[2\].set\_xlabel(f'PC1 ({variance\_expliquee\[0\]\*100:.1f}%)')

axes\[2\].set\_ylabel(f'PC2 ({variance\_expliquee\[1\]\*100:.1f}%)')

axes\[2\].set\_title('Projection PCA \- Clusters K-Means')

axes\[2\].grid(True, alpha=0.3)

plt.colorbar(scatter2, ax=axes\[2\])

plt.tight\_layout()

plt.show()

\# Biplot (contribution des variables)

print("\\nContribution des variables aux composantes principales:")

loadings \= pca.components\_.T \* np.sqrt(pca.explained\_variance\_)

loading\_df \= pd.DataFrame(

    loadings\[:, :2\],

    columns=\['PC1', 'PC2'\],

    index=X.columns

)

print(loading\_df)

\# Visualiser le biplot

plt.figure(figsize=(10, 8))

plt.scatter(X\_pca\[:, 0\], X\_pca\[:, 1\], alpha=0.3, s=30)

for i, feature in enumerate(X.columns):

    plt.arrow(0, 0, loadings\[i, 0\]\*3, loadings\[i, 1\]\*3,

              head\_width=0.1, head\_length=0.1, fc='red', ec='red')

    plt.text(loadings\[i, 0\]\*3.2, loadings\[i, 1\]\*3.2, feature,

             fontsize=12, fontweight='bold')

plt.xlabel(f'PC1 ({variance\_expliquee\[0\]\*100:.1f}%)')

plt.ylabel(f'PC2 ({variance\_expliquee\[1\]\*100:.1f}%)')

plt.title('Biplot PCA \- Variables et Observations')

plt.grid(True, alpha=0.3)

plt.axhline(y=0, color='k', linewidth=0.5)

plt.axvline(x=0, color='k', linewidth=0.5)

plt.show()

---

## 3.6 ÉVALUATION DES MODÈLES

### 3.6.1 Métriques de Performance pour la Classification

from sklearn.metrics import (accuracy\_score, precision\_score, recall\_score, 

                             f1\_score, matthews\_corrcoef, cohen\_kappa\_score)

print("=" \* 80\)

print("MÉTRIQUES D'ÉVALUATION \- CLASSIFICATION")

print("=" \* 80\)

\# Utiliser le modèle de régression logistique du Titanic

y\_pred \= log\_reg.predict(X\_test\_scaled)

y\_pred\_proba \= log\_reg.predict\_proba(X\_test\_scaled)\[:, 1\]

\# Calculer toutes les métriques

metriques \= {

    'Accuracy': accuracy\_score(y\_test, y\_pred),

    'Precision': precision\_score(y\_test, y\_pred),

    'Recall (Sensibilité)': recall\_score(y\_test, y\_pred),

    'F1-Score': f1\_score(y\_test, y\_pred),

    'MCC': matthews\_corrcoef(y\_test, y\_pred),

    'Cohen Kappa': cohen\_kappa\_score(y\_test, y\_pred),

}

print("\\nMÉTRIQUES DE PERFORMANCE:")

for metric, value in metriques.items():

    print(f"{metric:.\<30} {value:.4f}")

\# Matrice de confusion détaillée

tn, fp, fn, tp \= confusion\_matrix(y\_test, y\_pred).ravel()

print("\\nMATRICE DE CONFUSION DÉTAILLÉE:")

print(f"Vrais Négatifs (TN): {tn}")

print(f"Faux Positifs (FP): {fp}")

print(f"Faux Négatifs (FN): {fn}")

print(f"Vrais Positifs (TP): {tp}")

print("\\nMÉTRIQUES CALCULÉES MANUELLEMENT:")

print(f"Accuracy \= (TP \+ TN) / Total \= {(tp \+ tn) / (tp \+ tn \+ fp \+ fn):.4f}")

print(f"Precision \= TP / (TP \+ FP) \= {tp / (tp \+ fp):.4f}")

print(f"Recall \= TP / (TP \+ FN) \= {tp / (tp \+ fn):.4f}")

print(f"F1-Score \= 2 \* (Precision \* Recall) / (Precision \+ Recall)")

\# Courbe Precision-Recall

from sklearn.metrics import precision\_recall\_curve, average\_precision\_score

precision, recall, thresholds \= precision\_recall\_curve(y\_test, y\_pred\_proba)

ap\_score \= average\_precision\_score(y\_test, y\_pred\_proba)

fig, axes \= plt.subplots(1, 2, figsize=(14, 5))

\# Courbe Precision-Recall

axes\[0\].plot(recall, precision, linewidth=2, label=f'AP \= {ap\_score:.2f}')

axes\[0\].set\_xlabel('Recall')

axes\[0\].set\_ylabel('Precision')

axes\[0\].set\_title('Courbe Precision-Recall')

axes\[0\].legend()

axes\[0\].grid(True, alpha=0.3)

\# Impact du seuil

f1\_scores \= 2 \* (precision\[:-1\] \* recall\[:-1\]) / (precision\[:-1\] \+ recall\[:-1\])

axes\[1\].plot(thresholds, precision\[:-1\], label='Precision', linewidth=2)

axes\[1\].plot(thresholds, recall\[:-1\], label='Recall', linewidth=2)

axes\[1\].plot(thresholds, f1\_scores, label='F1-Score', linewidth=2)

axes\[1\].set\_xlabel('Seuil de décision')

axes\[1\].set\_ylabel('Score')

axes\[1\].set\_title('Impact du Seuil sur les Métriques')

axes\[1\].legend()

axes\[1\].grid(True, alpha=0.3)

plt.tight\_layout()

plt.show()

### 3.6.2 Métriques de Performance pour la Régression

from sklearn.metrics import mean\_absolute\_percentage\_error, max\_error

print("=" \* 80\)

print("MÉTRIQUES D'ÉVALUATION \- RÉGRESSION")

print("=" \* 80\)

\# Utiliser le modèle de régression des salaires

y\_pred\_test \= model.predict(X\_test)

\# Calculer les métriques

mae \= mean\_absolute\_error(y\_test, y\_pred\_test)

mse \= mean\_squared\_error(y\_test, y\_pred\_test)

rmse \= np.sqrt(mse)

r2 \= r2\_score(y\_test, y\_pred\_test)

mape \= mean\_absolute\_percentage\_error(y\_test, y\_pred\_test)

max\_err \= max\_error(y\_test, y\_pred\_test)

print("\\nMÉTRIQUES DE RÉGRESSION:")

print(f"MAE (Mean Absolute Error):........ {mae:.2f}")

print(f"MSE (Mean Squared Error):......... {mse:.2f}")

print(f"RMSE (Root Mean Squared Error):... {rmse:.2f}")

print(f"R² Score:......................... {r2:.4f}")

print(f"MAPE (Mean Abs Percentage Error):. {mape\*100:.2f}%")

print(f"Max Error:........................ {max\_err:.2f}")

print("\\nINTERPRÉTATION:")

print(f"- En moyenne, nos prédictions s'écartent de {mae:.2f} de la réalité")

print(f"- Le modèle explique {r2\*100:.2f}% de la variance des données")

print(f"- L'erreur relative moyenne est de {mape\*100:.2f}%")

### 3.6.3 Validation Croisée (Cross-Validation)

La validation croisée évalue la performance du modèle sur différents sous-ensembles des données.

from sklearn.model\_selection import cross\_val\_score, cross\_validate, KFold

print("=" \* 80\)

print("VALIDATION CROISÉE")

print("=" \* 80\)

\# K-Fold Cross-Validation

kfold \= KFold(n\_splits=5, shuffle=True, random\_state=42)

\# Classification avec Régression Logistique

scores \= cross\_val\_score(LogisticRegression(max\_iter=1000), 

                        X\_train\_scaled, y\_train, 

                        cv=kfold, scoring='accuracy')

print("\\nCLASSIFICATION \- Régression Logistique (5-fold CV):")

print(f"Scores par fold: {scores}")

print(f"Accuracy moyenne: {scores.mean():.4f} (+/- {scores.std() \* 2:.4f})")

\# Cross-validation avec plusieurs métriques

scoring \= \['accuracy', 'precision', 'recall', 'f1', 'roc\_auc'\]

cv\_results \= cross\_validate(LogisticRegression(max\_iter=1000),

                            X\_train\_scaled, y\_train,

                            cv=kfold, scoring=scoring)

print("\\nMÉTRIQUES MULTIPLES:")

for metric in scoring:

    scores \= cv\_results\[f'test\_{metric}'\]

    print(f"{metric:.\<20} {scores.mean():.4f} (+/- {scores.std() \* 2:.4f})")

\# Visualisation

plt.figure(figsize=(10, 6))

metrics\_means \= \[cv\_results\[f'test\_{m}'\].mean() for m in scoring\]

metrics\_stds \= \[cv\_results\[f'test\_{m}'\].std() for m in scoring\]

x\_pos \= np.arange(len(scoring))

plt.bar(x\_pos, metrics\_means, yerr=metrics\_stds, alpha=0.7, capsize=10)

plt.xticks(x\_pos, scoring)

plt.ylabel('Score')

plt.title('Performance avec Validation Croisée (5-fold)')

plt.ylim(\[0, 1\])

plt.grid(True, alpha=0.3, axis='y')

plt.show()

### 3.6.4 Surapprentissage et Sous-apprentissage

**Surapprentissage (Overfitting) :** Le modèle performe bien sur les données d'entraînement mais mal sur de nouvelles données.

**Sous-apprentissage (Underfitting) :** Le modèle ne capture pas les patterns des données.

from sklearn.model\_selection import learning\_curve

print("=" \* 80\)

print("ANALYSE SURAPPRENTISSAGE / SOUS-APPRENTISSAGE")

print("=" \* 80\)

\# Calculer les courbes d'apprentissage

train\_sizes, train\_scores, val\_scores \= learning\_curve(

    LogisticRegression(max\_iter=1000),

    X\_train\_scaled, y\_train,

    train\_sizes=np.linspace(0.1, 1.0, 10),

    cv=5,

    scoring='accuracy',

    random\_state=42

)

\# Calculer moyennes et écarts-types

train\_mean \= train\_scores.mean(axis=1)

train\_std \= train\_scores.std(axis=1)

val\_mean \= val\_scores.mean(axis=1)

val\_std \= val\_scores.std(axis=1)

\# Visualisation

plt.figure(figsize=(10, 6))

plt.plot(train\_sizes, train\_mean, label='Score Train', marker='o', linewidth=2)

plt.fill\_between(train\_sizes, train\_mean \- train\_std, train\_mean \+ train\_std, alpha=0.15)

plt.plot(train\_sizes, val\_mean, label='Score Validation', marker='s', linewidth=2)

plt.fill\_between(train\_sizes, val\_mean \- val\_std, val\_mean \+ val\_std, alpha=0.15)

plt.xlabel('Taille de l\\'ensemble d\\'entraînement')

plt.ylabel('Accuracy')

plt.title('Courbes d\\'Apprentissage \- Détection du Surapprentissage')

plt.legend(loc='best')

plt.grid(True, alpha=0.3)

plt.show()

print("\\nINTERPRÉTATION:")

gap \= train\_mean\[-1\] \- val\_mean\[-1\]

if gap \> 0.1:

    print(f"⚠️ SURAPPRENTISSAGE détecté (gap \= {gap:.4f})")

    print("Solutions: régularisation, plus de données, réduire la complexité")

elif val\_mean\[-1\] \< 0.7:

    print("⚠️ SOUS-APPRENTISSAGE détecté")

    print("Solutions: modèle plus complexe, plus de features, moins de régularisation")

else:

    print("✅ Le modèle semble bien équilibré")

\# Exemple de régularisation

print("\\n" \+ "="\*80)

print("IMPACT DE LA RÉGULARISATION")

print("="\*80)

C\_values \= \[0.001, 0.01, 0.1, 1, 10, 100\]

train\_scores \= \[\]

test\_scores \= \[\]

for C in C\_values:

    lr \= LogisticRegression(C=C, max\_iter=1000, random\_state=42)

    lr.fit(X\_train\_scaled, y\_train)

    train\_scores.append(lr.score(X\_train\_scaled, y\_train))

    test\_scores.append(lr.score(X\_test\_scaled, y\_test))

plt.figure(figsize=(10, 6))

plt.plot(C\_values, train\_scores, label='Train', marker='o', linewidth=2)

plt.plot(C\_values, test\_scores, label='Test', marker='s', linewidth=2)

plt.xscale('log')

plt.xlabel('Paramètre C (inverse de la régularisation)')

plt.ylabel('Accuracy')

plt.title('Impact de la Régularisation sur la Performance')

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()

print(f"\\nMeilleur C: {C\_values\[np.argmax(test\_scores)\]}")

---

## 3.7 OPTIMISATION DES HYPERPARAMÈTRES

### 3.7.1 Grid Search

Grid Search teste toutes les combinaisons possibles d'hyperparamètres.

from sklearn.model\_selection import GridSearchCV

print("=" \* 80\)

print("OPTIMISATION PAR GRID SEARCH")

print("=" \* 80\)

\# Définir la grille de paramètres

param\_grid \= {

    'C': \[0.001, 0.01, 0.1, 1, 10, 100\],

    'penalty': \['l1', 'l2'\],

    'solver': \['liblinear'\]

}

\# Créer le Grid Search

grid\_search \= GridSearchCV(

    LogisticRegression(max\_iter=1000, random\_state=42),

    param\_grid,

    cv=5,

    scoring='accuracy',

    n\_jobs=-1,

    verbose=1

)

\# Entraîner

print("\\nRecherche des meilleurs paramètres en cours...")

grid\_search.fit(X\_train\_scaled, y\_train)

\# Résultats

print("\\nMEILLEURS PARAMÈTRES:")

print(grid\_search.best\_params\_)

print(f"\\nMeilleur score (CV): {grid\_search.best\_score\_:.4f}")

print(f"Score sur test: {grid\_search.score(X\_test\_scaled, y\_test):.4f}")

\# Visualiser les résultats

results\_df \= pd.DataFrame(grid\_search.cv\_results\_)

pivot\_table \= results\_df.pivot\_table(

    values='mean\_test\_score',

    index='param\_C',

    columns='param\_penalty'

)

plt.figure(figsize=(10, 6))

sns.heatmap(pivot\_table, annot=True, fmt='.4f', cmap='YlGnBu')

plt.title('Grid Search \- Accuracy par Combinaison de Paramètres')

plt.xlabel('Penalty')

plt.ylabel('C')

plt.show()

### 3.7.2 Random Search

Random Search échantillonne aléatoirement l'espace des hyperparamètres.

from sklearn.model\_selection import RandomizedSearchCV

from scipy.stats import uniform, randint

print("=" \* 80\)

print("OPTIMISATION PAR RANDOM SEARCH")

print("=" \* 80\)

\# Définir les distributions de paramètres

param\_distributions \= {

    'n\_estimators': randint(50, 500),

    'max\_depth': randint(3, 20),

    'min\_samples\_split': randint(2, 20),

    'min\_samples\_leaf': randint(1, 10),

    'max\_features': \['sqrt', 'log2', None\]

}

\# Créer le Random Search

random\_search \= RandomizedSearchCV(

    RandomForestClassifier(random\_state=42),

    param\_distributions,

    n\_iter=50,  \# Nombre d'itérations

    cv=5,

    scoring='accuracy',

    random\_state=42,

    n\_jobs=-1,

    verbose=1

)

\# Entraîner

print("\\nRecherche aléatoire en cours...")

random\_search.fit(X\_train, y\_train)

\# Résultats

print("\\nMEILLEURS PARAMÈTRES:")

for param, value in random\_search.best\_params\_.items():

    print(f"  {param}: {value}")

print(f"\\nMeilleur score (CV): {random\_search.best\_score\_:.4f}")

print(f"Score sur test: {random\_search.score(X\_test, y\_test):.4f}")

\# Comparer modèle de base vs optimisé

rf\_base \= RandomForestClassifier(random\_state=42)

rf\_base.fit(X\_train, y\_train)

print("\\nCOMPARAISON:")

print(f"Modèle de base (test): {rf\_base.score(X\_test, y\_test):.4f}")

print(f"Modèle optimisé (test): {random\_search.score(X\_test, y\_test):.4f}")

print(f"Amélioration: {(random\_search.score(X\_test, y\_test) \- rf\_base.score(X\_test, y\_test))\*100:.2f}%")

---

## TRAVAUX PRATIQUES 2 : PROJET COMPLET DE MACHINE LEARNING

### Objectif

Développer un système complet de prédiction du churn client pour une entreprise de télécommunications.

### Dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model\_selection import train\_test\_split, cross\_val\_score, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear\_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification\_report, confusion\_matrix, roc\_auc\_score, roc\_curve

print("=" \* 80\)

print("PROJET ML: PRÉDICTION DU CHURN CLIENT \- TÉLÉCOMMUNICATIONS")

print("=" \* 80\)

\# Charger les données

url \= "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df\_churn \= pd.read\_csv(url)

print("\\n1. EXPLORATION DES DONNÉES")

print("-" \* 80\)

print(f"Shape: {df\_churn.shape}")

print(f"\\nPremières lignes:")

print(df\_churn.head())

\# Informations sur les colonnes

print(f"\\nInfo:")

df\_churn.info()

\# Statistiques descriptives

print(f"\\nStatistiques:")

print(df\_churn.describe())

\# Distribution du churn

print(f"\\nDistribution du Churn:")

print(df\_churn\['Churn'\].value\_counts(normalize=True))

print("\\n2. NETTOYAGE DES DONNÉES")

print("-" \* 80\)

\# Convertir TotalCharges en numérique

df\_churn\['TotalCharges'\] \= pd.to\_numeric(df\_churn\['TotalCharges'\], errors='coerce')

\# Gérer les valeurs manquantes

print(f"Valeurs manquantes:")

print(df\_churn.isnull().sum()\[df\_churn.isnull().sum() \> 0\])

df\_churn\['TotalCharges'\].fillna(df\_churn\['TotalCharges'\].median(), inplace=True)

\# Supprimer customerID

df\_churn.drop('customerID', axis=1, inplace=True)

print("\\n3. FEATURE ENGINEERING")

print("-" \* 80\)

\# Créer de nouvelles features

df\_churn\['ChargePerMonth'\] \= df\_churn\['TotalCharges'\] / (df\_churn\['tenure'\] \+ 1\)

df\_churn\['HasMultipleServices'\] \= (

    (df\_churn\['OnlineSecurity'\] \== 'Yes') | 

    (df\_churn\['OnlineBackup'\] \== 'Yes') | 

    (df\_churn\['DeviceProtection'\] \== 'Yes')

).astype(int)

print("Nouvelles features créées: ChargePerMonth, HasMultipleServices")

print("\\n4. ENCODAGE DES VARIABLES")

print("-" \* 80\)

\# Variables catégorielles binaires

binary\_cols \= \['gender', 'Partner', 'Dependents', 'PhoneService', 

               'PaperlessBilling', 'Churn'\]

le \= LabelEncoder()

for col in binary\_cols:

    df\_churn\[f'{col}\_encoded'\] \= le.fit\_transform(df\_churn\[col\])

\# Variables catégorielles multi-classes

multi\_cols \= \['InternetService', 'Contract', 'PaymentMethod'\]

df\_churn \= pd.get\_dummies(df\_churn, columns=multi\_cols, drop\_first=True)

print(f"Nombre de colonnes après encodage: {df\_churn.shape\[1\]}")

print("\\n5. PRÉPARATION POUR LA MODÉLISATION")

print("-" \* 80\)

\# Sélectionner les features

feature\_cols \= \[col for col in df\_churn.columns 

                if col not in \['gender', 'Partner', 'Dependents', 'PhoneService',

                               'PaperlessBilling', 'Churn', 'MultipleLines',

                               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                               'TechSupport', 'StreamingTV', 'StreamingMovies'\]\]

X \= df\_churn\[feature\_cols\]

y \= df\_churn\['Churn\_encoded'\]

print(f"Features: {len(feature\_cols)}")

print(f"Observations: {len(X)}")

\# Séparer train/test

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(

    X, y, test\_size=0.2, random\_state=42, stratify=y

)

\# Standardiser

scaler \= StandardScaler()

X\_train\_scaled \= scaler.fit\_transform(X\_train)

X\_test\_scaled \= scaler.transform(X\_test)

print(f"Train set: {X\_train.shape}")

print(f"Test set: {X\_test.shape}")

print("\\n6. ENTRAÎNEMENT DE PLUSIEURS MODÈLES")

print("-" \* 80\)

\# Dictionnaire de modèles

models \= {

    'Logistic Regression': LogisticRegression(max\_iter=1000, random\_state=42),

    'Decision Tree': DecisionTreeClassifier(random\_state=42),

    'Random Forest': RandomForestClassifier(n\_estimators=100, random\_state=42),

    'Gradient Boosting': GradientBoostingClassifier(random\_state=42)

}

\# Entraîner et évaluer chaque modèle

results \= {}

for name, model in models.items():

    print(f"\\nEntraînement: {name}")

    

    \# Entraîner

    if name \== 'Logistic Regression':

        model.fit(X\_train\_scaled, y\_train)

        y\_pred \= model.predict(X\_test\_scaled)

        y\_pred\_proba \= model.predict\_proba(X\_test\_scaled)\[:, 1\]

    else:

        model.fit(X\_train, y\_train)

        y\_pred \= model.predict(X\_test)

        y\_pred\_proba \= model.predict\_proba(X\_test)\[:, 1\]

    

    \# Métriques

    from sklearn.metrics import accuracy\_score, precision\_score, recall\_score, f1\_score

    

    results\[name\] \= {

        'Accuracy': accuracy\_score(y\_test, y\_pred),

        'Precision': precision\_score(y\_test, y\_pred),

        'Recall': recall\_score(y\_test, y\_pred),

        'F1-Score': f1\_score(y\_test, y\_pred),

        'ROC-AUC': roc\_auc\_score(y\_test, y\_pred\_proba)

    }

    

    print(f"  Accuracy: {results\[name\]\['Accuracy'\]:.4f}")

    print(f"  ROC-AUC: {results\[name\]\['ROC-AUC'\]:.4f}")

\# Tableau récapitulatif

results\_df \= pd.DataFrame(results).T

print("\\n7. COMPARAISON DES MODÈLES")

print("-" \* 80\)

print(results\_df)

\# Visualiser les résultats

fig, axes \= plt.subplots(1, 2, figsize=(161 Introduction au Machine Learning

Le \*\*Machine Learning (ML)\*\* est une branche de l'intelligence artificielle qui permet aux ordinateurs d'apprendre à partir de données sans être explicitement programmés.

\*\*Différence avec la programmation traditionnelle :\*\*

| Programmation Traditionnelle | Machine Learning |

|------------------------------|------------------|

| Règles explicites définies par le développeur | Règles apprises automatiquement |

| Logique if-then-else | Apprentissage par patterns |

| Difficile à adapter | S'améliore avec plus de données |

| Exemple : Calcul de TVA | Exemple : Détection de spam |

\*\*Types d'apprentissage :\*\*

1\. \*\*Apprentissage Supervisé\*\* : Apprendre à partir de données étiquetées

   \- Régression : Prédire une valeur continue

   \- Classification : Prédire une catégorie

2\. \*\*Apprentissage Non Supervisé\*\* : Découvrir des structures dans des données non étiquetées

   \- Clustering : Regrouper des observations similaires

   \- Réduction de dimensionnalité : Simplifier les données

3\. \*\*Apprentissage par Renforcement\*\* : Apprendre par essai-erreur (hors programme)

\#\#\# 3.2 Préparation des Données pour le ML

\*\*Workflow typique :\*\*

\`\`\`python

import pandas as pd

import numpy as np

from sklearn.model\_selection import train\_test\_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

\# Charger des données

url \= "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df \= pd.read\_csv(url)

print("=" \* 80\)

print("PRÉPARATION DES DONNÉES POUR LE MACHINE LEARNING")

print("=" \* 80\)

\# 1\. EXPLORATION INITIALE

print("\\n1. APERÇU DES DONNÉES")

print("-" \* 80\)

print(df.head())

print(f"\\nDimensions: {df.shape}")

print(f"\\nTypes de données:\\n{df.dtypes}")

\# 2\. GESTION DES VALEURS MANQUANTES

print("\\n2. VALEURS MANQUANTES")

print("-" \* 80\)

print(df.isnull().sum())

\# Stratégies de traitement

\# a) Imputation par la médiane pour Age

df\['Age'\].fillna(df\['Age'\].median(), inplace=True)

\# b) Imputation par le mode pour Embarked

df\['Embarked'\].fillna(df\['Embarked'\].mode()\[0\], inplace=True)

\# c) Supprimer Cabin (trop de valeurs manquantes)

df.drop('Cabin', axis=1, inplace=True)

print("\\nAprès traitement:")

print(df.isnull().sum())

\# 3\. ENCODAGE DES VARIABLES CATÉGORIELLES

print("\\n3. ENCODAGE DES VARIABLES CATÉGORIELLES")

print("-" \* 80\)

\# Label Encoding pour variables binaires

le \= LabelEncoder()

df\['Sex\_encoded'\] \= le.fit\_transform(df\['Sex'\])

\# One-Hot Encoding pour variables avec plusieurs catégories

df \= pd.get\_dummies(df, columns=\['Embarked'\], prefix='Embarked')

print("Variables encodées:")

print(df\[\['Sex', 'Sex\_encoded'\]\].head())

print(df\[\[col for col in df.columns if 'Embarked' in col\]\].head())

\# 4\. FEATURE ENGINEERING

print("\\n4. FEATURE ENGINEERING")

print("-" \* 80\)

\# Créer de nouvelles features

df\['FamilySize'\] \= df\['SibSp'\] \+ df\['Parch'\] \+ 1

df\['IsAlone'\] \= (df\['FamilySize'\] \== 1).astype(int)

df\['Title'\] \= df\['Name'\].str.extract(' (\[A-Za-z\]+)\\.', expand=False)

print("Nouvelles features créées:")

print(df\[\['SibSp', 'Parch', 'FamilySize', 'IsAlone'\]\].head())

print(f"\\nTitres uniques: {df\['Title'\].unique()}")

\# 5\. SÉLECTION DES FEATURES

print("\\n5. SÉLECTION DES FEATURES")

print("-" \* 80\)

\# Features pour la modélisation

features \= \['Pclass', 'Sex\_encoded', 'Age', 'SibSp', 'Parch', 

            'Fare', 'FamilySize', 'IsAlone'\] \+ \\

           \[col for col in df.columns if 'Embarked' in col\]

X \= df\[features\]

y \= df\['Survived'\]

print(f"Features sélectionnées: {features}")

print(f"Shape X: {X.shape}, Shape y: {y.shape}")

\# 6\. SÉPARATION TRAIN/TEST

print("\\n6. SÉPARATION DONNÉES D'ENTRAÎNEMENT/TEST")

print("-" \* 80\)

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(

    X, y, test\_size=0.2, random\_state=42, stratify=y

)

print(f"Train set: {X\_train.shape}, {y\_train.shape}")

print(f"Test set: {X\_test.shape}, {y\_test.shape}")

print(f"\\nRépartition des classes dans train:")

print(y\_train.value\_counts(normalize=True))

\# 7\. NORMALISATION/STANDARDISATION

print("\\n7. NORMALISATION DES DONNÉES")

print("-" \* 80\)

scaler \= StandardScaler()

X\_train\_scaled \= scaler.fit\_transform(X\_train)

X\_test\_scaled \= scaler.transform(X\_test)

print("Statistiques avant standardisation:")

print(f"Moyenne: {X\_train\['Age'\].mean():.2f}, Std: {X\_train\['Age'\].std():.2f}")

print("\\nStatistiques après standardisation:")

print(f"Moyenne: {X\_train\_scaled\[:, 2\].mean():.2f}, Std: {X\_train\_scaled\[:, 2\].std():.2f}")

print("\\n" \+ "=" \* 80\)

print("DONNÉES PRÊTES POUR LA MODÉLISATION")

print("=" \* 80\)

---

## 3.3 APPRENTISSAGE SUPERVISÉ \- RÉGRESSION

### 3.3.1 Régression Linéaire Simple

La régression linéaire modélise la relation entre une variable dépendante $y$ et une variable indépendante $x$ :

$$y \= \\beta\_0 \+ \\beta\_1 x \+ \\epsilon$$

Où :

- $\\beta\_0$ : ordonnée à l'origine (intercept)  
- $\\beta\_1$ : pente (coefficient)  
- $\\epsilon$ : erreur

**Cas d'usage en entreprise :**

- Prédire les ventes en fonction du budget marketing  
- Estimer le salaire en fonction de l'expérience  
- Prévoir le chiffre d'affaires

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import mean\_squared\_error, r2\_score, mean\_absolute\_error

import numpy as np

\# Charger des données de salaires

url \= "https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Salary%20Prediction/Salary\_Data.csv"

df\_salary \= pd.read\_csv(url)

print("=" \* 80\)

print("RÉGRESSION LINÉAIRE SIMPLE \- PRÉDICTION DE SALAIRE")

print("=" \* 80\)

\# Préparer les données

X \= df\_salary\[\['YearsExperience'\]\].values

y \= df\_salary\['Salary'\].values

\# Séparer train/test

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(

    X, y, test\_size=0.2, random\_state=42

)

\# Créer et entraîner le modèle

model \= LinearRegression()

model.fit(X\_train, y\_train)

\# Faire des prédictions

y\_pred\_train \= model.predict(X\_train)

y\_pred\_test \= model.predict(X\_test)

\# Évaluer le modèle

print("\\nPARAMÈTRES DU MODÈLE:")

print(f"Intercept (β₀): {model.intercept\_:.2f}")

print(f"Coefficient (β₁): {model.coef\_\[0\]:.2f}")

print(f"\\nÉquation: Salaire \= {model.intercept\_:.2f} \+ {model.coef\_\[0\]:.2f} × Années")

print("\\nPERFORMANCE SUR TRAIN:")

print(f"R² Score: {r2\_score(y\_train, y\_pred\_train):.4f}")

print(f"RMSE: {np.sqrt(mean\_squared\_error(y\_train, y\_pred\_train)):.2f}")

print(f"MAE: {mean\_absolute\_error(y\_train, y\_pred\_train):.2f}")

print("\\nPERFORMANCE SUR TEST:")

print(f"R² Score: {r2\_score(y\_test, y\_pred\_test):.4f}")

print(f"RMSE: {np.sqrt(mean\_squared\_error(y\_test, y\_pred\_test)):.2f}")

print(f"MAE: {mean\_absolute\_error(y\_test, y\_pred\_test):.2f}")

\# Visualisation

fig, axes \= plt.subplots(1, 2, figsize=(14, 5))

\# Graphique 1: Données et ligne de régression

axes\[0\].scatter(X\_train, y\_train, alpha=0.6, label='Train', color='blue')

axes\[0\].scatter(X\_test, y\_test, alpha=0.6, label='Test', color='green')

axes\[0\].plot(X, model.predict(X), color='red', linewidth=2, label='Régression')

axes\[0\].set\_xlabel('Années d\\'expérience')

axes\[0\].set\_ylabel('Salaire')

axes\[0\].set\_title('Régression Linéaire: Salaire vs Expérience')

axes\[0\].legend()

axes\[0\].grid(True, alpha=0.3)

\# Graphique 2: Résidus

residus \= y\_test \- y\_pred\_test

axes\[1\].scatter(y\_pred\_test, residus, alpha=0.6)

axes\[1\].axhline(y=0, color='red', linestyle='--', linewidth=2)

axes\[1\].set\_xlabel('Valeurs prédites')

axes\[1\].set\_ylabel('Résidus')

axes\[1\].set\_title('Analyse des Résidus')

axes\[1\].grid(True, alpha=0.3)

plt.tight\_layout()

plt.show()

\# Faire une prédiction pour un nouveau cas

nouvelle\_experience \= np.array(\[\[5.0\]\])

salaire\_predit \= model.predict(nouvelle\_experience)

print(f"\\nPRÉDICTION: Avec 5 ans d'expérience, salaire estimé: {salaire\_predit\[0\]:.2f}")

### 3.3.2 Régression Linéaire Multiple

La régression multiple utilise plusieurs variables indépendantes :

$$y \= \\beta\_0 \+ \\beta\_1 x\_1 \+ \\beta\_2 x\_2 \+ ... \+ \\beta\_n x\_n \+ \\epsilon$$

\# Charger des données immobilières

url \= "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

df\_housing \= pd.read\_csv(url)

print("=" \* 80\)

print("RÉGRESSION LINÉAIRE MULTIPLE \- PRIX IMMOBILIERS")

print("=" \* 80\)

print("\\nVariables disponibles:")

print(df\_housing.columns.tolist())

\# Sélectionner les features

features \= \['crim', 'rm', 'age', 'dis', 'tax', 'ptratio', 'lstat'\]

X \= df\_housing\[features\]

y \= df\_housing\['medv'\]

\# Séparer et standardiser

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(

    X, y, test\_size=0.2, random\_state=42

)

scaler \= StandardScaler()

X\_train\_scaled \= scaler.fit\_transform(X\_train)

X\_test\_scaled \= scaler.transform(X\_test)

\# Entraîner le modèle

model\_multi \= LinearRegression()

model\_multi.fit(X\_train\_scaled, y\_train)

\# Prédictions

y\_pred\_train \= model\_multi.predict(X\_train\_scaled)

y\_pred\_test \= model\_multi.predict(X\_test\_scaled)

\# Évaluation

print("\\nPERFORMANCE:")

print(f"R² Train: {r2\_score(y\_train, y\_pred\_train):.4f}")

print(f"R² Test: {r2\_score(y\_test, y\_pred\_test):.4f}")

print(f"RMSE Test: {np.sqrt(mean\_squared\_error(y\_test, y\_pred\_test)):.2f}")

\# Importance des features

importance \= pd.DataFrame({

    'Feature': features,

    'Coefficient': model\_multi.coef\_

}).sort\_values('Coefficient', key=abs, ascending=False)

print("\\nIMPORTANCE DES VARIABLES:")

print(importance)

\# Visualisation

fig, axes \= plt.subplots(1, 2, figsize=(14, 5))

\# Valeurs réelles vs prédites

axes\[0\].scatter(y\_test, y\_pred\_test, alpha=0.5)

axes\[0\].plot(\[y\_test.min(), y\_test.max()\], 

             \[y\_test.min(), y\_test.max()\], 

             'r--', linewidth=2)

axes\[0\].set\_xlabel('Prix réel')

axes\[0\].set\_ylabel('Prix prédit')

axes\[0\].set\_title('Prédictions vs Réalité')

axes\[0\].grid(True, alpha=0.3)

\# Importance des coefficients

axes\[1\].barh(importance\['Feature'\], np.abs(importance\['Coefficient'\]))

axes\[1\].set\_xlabel('|Coefficient|')

axes\[1\].set\_title('Importance des Variables')

axes\[1\].grid(True, alpha=0.3)

plt.tight\_layout()

plt.show()

### 3.3.3 Régression Logistique

Malgré son nom, la régression logistique est utilisée pour la **classification binaire**.

**Fonction sigmoïde :**

$$P(y=1|x) \= \\frac{1}{1 \+ e^{-(\\beta\_0 \+ \\beta\_1 x)}}$$

**Cas d'usage :**

- Prédire le churn client (partir/rester)  
- Détection de fraude (frauduleux/légitime)  
- Scoring de crédit (accepter/rejeter)

from sklearn.linear\_model import LogisticRegression

from sklearn.metrics import classification\_report, confusion\_matrix, roc\_curve, auc

\# Utiliser les données Titanic préparées précédemment

print("=" \* 80\)

print("RÉGRESSION LOGISTIQUE \- PRÉDICTION DE SURVIE TITANIC")

print("=" \* 80\)

\# Créer et entraîner le modèle

log\_reg \= LogisticRegression(random\_state=42, max\_iter=1000)

log\_reg.fit(X\_train\_scaled, y\_train)

\# Prédictions

y\_pred\_train \= log\_reg.predict(X\_train\_scaled)

y\_pred\_test \= log\_reg.predict(X\_test\_scaled)

y\_pred\_proba \= log\_reg.predict\_proba(X\_test\_scaled)\[:, 1\]

\# Évaluation

print("\\nPERFORMANCE SUR TEST:")

print(f"Accuracy: {log\_reg.score(X\_test\_scaled, y\_test):.4f}")

print("\\nMATRICE DE CONFUSION:")

cm \= confusion\_matrix(y\_test, y\_pred\_test)

print(cm)

print("\\nRAPPORT DE CLASSIFICATION:")

print(classification\_report(y\_test, y\_pred\_test, 

                          target\_names=\['Décédé', 'Survécu'\]))

\# Visualisation

fig, axes \= plt.subplots(1, 2, figsize=(14, 5))

\# Matrice de confusion

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes\[0\],

            xticklabels=\['Décédé', 'Survécu'\],

            yticklabels=\['Décédé', 'Survécu'\])

axes\[0\].set\_ylabel('Réalité')

axes\[0\].set\_xlabel('Prédiction')

axes\[0\].set\_title('Matrice de Confusion')

\# Courbe ROC

fpr, tpr, thresholds \= roc\_curve(y\_test, y\_pred\_proba)

roc\_auc \= auc(fpr, tpr)

axes\[1\].plot(fpr, tpr, color='darkorange', linewidth=2,

             label=f'ROC curve (AUC \= {roc\_auc:.2f})')

axes\[1\].plot(\[0, 1\], \[0, 1\], color='navy', linewidth=2, linestyle='--')

axes\[1\].set\_xlim(\[0.0, 1.0\])

axes\[1\].set\_ylim(\[0.0, 1.05\])

axes\[1\].set\_xlabel('Taux de Faux Positifs')

axes\[1\].set\_ylabel('Taux de Vrais Positifs')

axes\[1\].set\_title('Courbe ROC')

axes\[1\].legend(loc="lower right")

axes\[1\].grid(True, alpha=0.3)

plt.tight\_layout()

plt.show()

\# Interprétation des coefficients

feature\_importance \= pd.DataFrame({

    'Feature': features,

    'Coefficient': log\_reg.coef\_\[0\]

}).sort\_values('Coefficient', ascending=False)

print("\\nIMPACT DES VARIABLES SUR LA SURVIE:")

print(feature\_importance)

---

## 3.4 APPRENTISSAGE SUPERVISÉ \- CLASSIFICATION

### 3.4.1 K-Nearest Neighbors (KNN)

KNN classifie un point en fonction des $k$ plus proches voisins.

**Principe :**

1. Calculer la distance entre le point à classer et tous les points d'entraînement  
2. Sélectionner les $k$ voisins les plus proches  
3. Attribuer la classe majoritaire parmi ces voisins

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy\_score

\# Charger des données de vin

url \= "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

columns \= \['Class', 'Alcohol', 'Malic\_acid', 'Ash', 'Alcalinity', 'Magnesium',

           'Phenols', 'Flavanoids', 'Nonflavanoid', 'Proanthocyanins',

           'Color\_intensity', 'Hue', 'OD280', 'Proline'\]

df\_wine \= pd.read\_csv(url, names=columns)

print("=" \* 80\)

print("K-NEAREST NEIGHBORS \- CLASSIFICATION DE VINS")

print("=" \* 80\)

\# Préparer les données

X \= df\_wine.drop('Class', axis=1)

y \= df\_wine\['Class'\]

X\_train, X\_test, y\_train, y\_test \= train\_test\_split(

    X, y, test\_size=0.2, random\_state=42, stratify=y

)

\# Standardiser (important pour KNN\!)

scaler \= StandardScaler()

X\_train\_scaled \= scaler.fit\_transform(X\_train)

X\_test\_scaled \= scaler.transform(X\_test)

\# Trouver le meilleur k

k\_values \= range(1, 21\)

train\_scores \= \[\]

test\_scores \= \[\]

for k in k\_values:

    knn \= KNeighborsClassifier(n\_neighbors=k)

    knn.fit(X\_train\_scaled, y\_train)

    train\_scores.append(knn.score(X\_train\_scaled, y\_train))

    test\_scores.append(knn.score(X\_test\_scaled, y\_test))

\# Visualiser les performances

plt.figure(figsize=(10, 6))

plt.plot(k\_values, train\_scores, label='Train', marker='o')

plt.plot(k\_values, test\_scores, label='Test', marker='s')

plt.xlabel('Nombre de voisins (k)')

plt.ylabel('Accuracy')

plt.title('Performance du KNN en fonction de k')

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()

\# Meilleur k

best\_k \= k\_values\[np.argmax(test\_scores)\]

print(f"\\nMeilleur k: {best\_k}")

\# Entraîner avec le meilleur k

knn\_best \= KNeighborsClassifier(n\_neighbors=best\_k)

knn\_best.fit(X\_train\_scaled, y\_train)

\# Évaluation

y\_pred \= knn\_best.predict(X\_test\_scaled)

print(f"Accuracy: {accuracy\_score(y\_test, y\_pred):.4f}")

print("\\nRAPPORT DE CLASSIFICATION:")

print(classification\_report(y\_test, y\_pred))

### 3.4.2 Support Vector Machine (SVM)

SVM trouve l'hyperplan optimal qui sépare les classes avec la marge maximale.

**Cas d'usage :**

- Classification de textes  
- Détection d'anomalies  
- Reconnaissance de patterns

from sklearn.svm import SVC

print("=" \* 80\)

print("SUPPORT VECTOR MACHINE \- CLASSIFICATION")

print("=" \* 80\)

\# Tester différents noyaux

kernels \= \['linear', 'rbf', 'poly'\]

results \= {}

for kernel in kernels:

    svm \= SVC(kernel=kernel, random\_state=42)

    svm.fit(X\_train\_scaled, y\_train)

    train\_score \= svm.score(X\_train\_scaled, y\_train)

    test\_score \= svm.score(X\_test\_scaled, y\_test)

    results\[kernel\] \= {'train': train\_score, 'test': test\_score}

    print(f"\\nKernel: {kernel}")

    print(f"  Train Accuracy: {train\_score:.4f}")

    print(f"  Test Accuracy: {test\_score:.4f}")

\# Meilleur modèle

best\_kernel \= max(results, key=lambda k: results\[k\]\['test'\])

print(f"\\nMeilleur noyau: {best\_kernel}")

\# Visualisation

kernels\_list \= list(results.keys())

train\_accs \= \[results\[k\]\['train'\] for k in kernels\_list\]

test\_accs \= \[results\[k\]\['test'\] for k in kernels\_list\]

x \= np.arange(len(kernels\_list))

width \= 0.35

fig, ax \= plt.subplots(figsize=(10, 6))

ax.bar(x \- width/2, train\_accs, width, label='Train')

ax.bar(x \+ width/2, test\_accs, width, label='Test')

ax.set\_ylabel('Accuracy')

ax.set\_title('Performance SVM par type de noyau')

ax.set\_xticks(x)

ax.set\_xticklabels(kernels\_list)

ax.legend()

ax.grid(True, alpha=0.3, axis='y')

plt.show()

### 3.4.3 Arbres de Décision et Forêts Aléatoires

**Arbre de Décision :** Structure hiérarchique de décisions basées sur les features.

**Forêt Aléatoire :** Ensemble d'arbres de décision qui votent pour la prédiction finale.

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot\_tree

print("=" \* 80\)

print("ARBRES DE DÉCISION ET FORÊTS ALÉATOIRES")

print("=" \* 80\)

\# Arbre de Décision

dt \= DecisionTreeClassifier(max\_depth=5, random\_state=42)

dt.fit(X\_train, y\_train)

\# Forêt Aléatoire

rf \= RandomForestClassifier(n\_estimators=100, random\_state=42)

rf.fit(X\_train, y\_train)

\# Évaluation

print("\\nARBRE DE DÉCISION:")

print(f"Train Accuracy: {dt.score(X\_train, y\_train):.4f}")

print(f"Test Accuracy: {dt.score(X\_test, y\_test):.4f}")

print("\\nFORÊT ALÉATOIRE:")

print(f"Train Accuracy: {rf.score(X\_train, y\_train):.4f}")

print(f"Test Accuracy: {rf.score(X\_test, y\_test):.4f}")

\# Importance des features

feature\_importance \= pd.DataFrame({

    'Feature': X.columns,

    'Importance': rf.feature\_importances\_

}).sort\_values('Importance', ascending=False)

print("\\nIMPORTANCE DES VARIABLES (Forêt Aléatoire):")

print(feature\_importance.head(10))

\# Visualisation

fig, axes \= plt.subplots(1, 2, figsize=(16, 6))

\# Arbre de décision

plot\_tree(dt, feature\_names=X.columns, class\_names=\[str(c) for c in y.unique()\],

          filled=True, ax=axes\[0\], fontsize=8)

axes\[0\].set\_title('Arbre de Décision (profondeur=5)')

\# Importance des features

top\_features \= feature\_importance.head(10)

axes\[1\].barh(range(len(top\_features)), top\_features\['Importance'\])

axes\[1\].set\_yticks(range(len(top\_features)))

axes\[1\].set\_yticklabels(top\_features\['Feature'\])

axes\[1\].set\_xlabel('Importance')

axes\[1\].set\_title('Top 10 Features \- Forêt Aléatoire')

axes\[1\].invert\_yaxis()

axes\[1\].grid(True, alpha=0.3, axis='x')

plt.tight\_layout()

plt.show()

---

## 3.5 APPRENTISSAGE NON SUPERVISÉ

### 3.5.1 Clustering : K-Means

K-Means regroupe les données en $k$ clusters en minimisant la variance intra-cluster.

**Cas d'usage en entreprise :**

- Segmentation client (RFM)  
- Segmentation produits  
- Détection d'anomalies

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette\_score

\# Charger des données clients

url \= "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

df\_iris \= pd.read\_csv(url)

print("=" \* 80\)

print("CLUSTERING K-MEANS \- SEGMENTATION")

print("=" \* 80\)

\# Préparer les données

X \= df\_iris.drop('species', axis=1)

\# Standardiser

scaler \= StandardScaler()

X\_scaled \= scaler.fit\_transform(X)

\# Méthode du coude pour trouver k optimal

inertias \= \[\]

silhouette\_scores \= \[\]

k\_range \= range(2, 11\)

for k in k\_range:

    kmeans \= KMeans(n\_clusters=k, random\_state=42, n\_init=10)

    kmeans.fit(X\_scaled)

    inertias.append(kmeans.inertia\_)

    silhouette\_scores.append(silhouette\_score(X\_scaled, kmeans.labels\_))

\# Visualisation

fig, axes \= plt.subplots(1, 2, figsize=(14, 5))

\# Méthode du coude

axes\[0\].plot(k\_range, inertias, marker='o', linewidth=2)

axes\[0\].set\_xlabel('Nombre de clusters (k)')

axes\[0\].set\_ylabel('Inertie')

axes\[0\].set\_title('Méthode du Coude')

axes\[0\].grid(True, alpha=0.3)

\# Score de silhouette

axes\[1\].plot(k\_range, silhouette\_scores, marker='s', linewidth=2, color='green')

axes\[1\].set\_xlabel('Nombre de clusters (k)')

axes\[1\].set\_ylabel('Score de Silhouette')

axes\[1\].set\_title('Score de Silhouette par k')

axes\[1\].grid(True, alpha=0.3)

plt.tight\_layout()

plt.show()

\# Choisir k=3 (on sait qu'il y a 3 espèces)

kmeans\_final \= KMeans(n\_clusters=3, random\_state=42, n\_init=10)

clusters \= kmeans\_final.fit\_predict(X\_scaled)

\# Ajouter les clusters au dataframe

df\_iris\['Cluster'\] \= clusters

print(f"\\nNombre d'observations par cluster:")

print(df\_iris\['Cluster'\].value\_counts().sort\_index())

\# Caractériser les clusters

print("\\nCaractéristiques moyennes par cluster:")

cluster\_profiles \= df\_iris.groupby('Cluster')\[X.columns\].mean()

print(cluster\_profiles)

\# Visualisation 2D (2 premières features)

plt.figure(figsize=(10, 6))

scatter \= plt.scatter(X\_scaled\[:, 0\], X\_scaled\[:, 1\], 

                     c=clusters, cmap='viridis', alpha=0.6, s=50)

plt.scatter(kmeans\_final.cluster\_centers\_\[:, 0\], 

           kmeans\_final.cluster\_centers\_\[:, 1\],

           c='red', marker='X', s=200, edgecolors='black', linewidths=2,

           label='Centroïdes')

plt.xlabel(X.columns\[0\])

plt.ylabel(X.columns\[1\])

plt.title('Clustering K-Means (3 clusters)')

plt.colorbar(scatter, label='Cluster')

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()

### 3\.

