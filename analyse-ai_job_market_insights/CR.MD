## EL Fadili Safae 

## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année


<img src="safae el fadili_photo.jpg" style="height:264px;margin-right:264px"/>

## Apogee: 24010411





















## Analyse du Marché de l'Emploi de l'IA et Prédiction des Salaires
## Table des Matières
Introduction du Projet
Analyse Exploratoire des Données (EDA)
Distribution de Salary_USD
Analyse des Relations entre les Variables Indépendantes et Salary_USD
Analyse de Corrélation
Préparation des Données pour la Modélisation
Modélisation de Régression - Linéaire Simple
Modélisation de Régression - Polynomiale
Modélisation de Régression - Ridge
Modélisation de Régression - Arbre de Décision
Explication sur la non-pertinence de la Régression Logistique
Comparaison Finale des Modèles et Conclusion Générale

## Introduction du Projet
Le jeu de données "AI-Powered Job Market Insights" offre un aperçu synthétique mais réaliste du marché du travail moderne, en se concentrant particulièrement sur le rôle de l'intelligence artificielle (IA) et de l'automatisation dans diverses industries. Ce jeu de données comprend 500 annonces d'emploi uniques, chacune caractérisée par différents facteurs tels que l'industrie, la taille de l'entreprise, le niveau d'adoption de l'IA, le risque d'automatisation, les compétences requises et les projections de croissance de l'emploi.
Il est conçu pour être une ressource précieuse pour les chercheurs, les data scientists et les décideurs politiques qui explorent l'impact de l'IA sur l'emploi, les tendances du marché du travail et l'avenir du travail.
# Analyse du Marché de l'Emploi en IA et Prédiction des Salaires

## Description du Jeu de Données

Le jeu de données contient des informations sur divers aspects du marché du travail, notamment:

*   **Job_Title**: Le titre du poste.
*   **Industry**: L'industrie à laquelle le poste appartient.
*   **Company_Size**: La taille de l'entreprise (petite, moyenne, grande).
*   **Location**: La localisation géographique du poste.
*   **AI_Adoption_Level**: Le niveau d'adoption de l'IA par l'entreprise.
*   **Automation_Risk**: Le risque d'automatisation du poste.
*   **Required_Skills**: Les compétences requises pour le poste.
*   **Salary_USD**: Le salaire annuel en dollars américains (variable cible).
*   **Remote_Friendly**: Indique si le poste est télétravaillable.
*   **Job_Growth_Projection**: La projection de croissance de l'emploi pour ce rôle.

## Objectif de l'Analyse

L'objectif principal de cette analyse est de:

1.  Comprendre les relations entre les différentes caractéristiques du marché de l'emploi (titre du poste, industrie, taille de l'entreprise, etc.) et le `Salary_USD`.
2.  Construire et évaluer différents modèles de régression pour prédire le `Salary_USD` en fonction de ces caractéristiques.

## Méthodologie

La méthodologie adoptée pour cette analyse comprend les étapes suivantes:

1.  **Chargement et Inspection des Données**: Lecture du jeu de données et examen initial de sa structure et de ses premières lignes.
2.  **Prétraitement des Données**: Identification et gestion des valeurs manquantes, des doublons, et encodage des caractéristiques catégorielles en valeurs numériques pour les rendre utilisables par les modèles.
3.  **Analyse Exploratoire des Données (EDA)**: Visualisation des distributions des variables clés et de leurs relations, y compris la distribution du `Salary_USD` et ses relations avec les autres caractéristiques.
4.  **Analyse de Corrélation**: Calcul et visualisation de la matrice de corrélation pour identifier les relations linéaires entre les variables.
5.  **Modélisation Prédictive**: Entraînement et évaluation de plusieurs modèles de régression:
    *   Régression Linéaire Simple
    *   Régression Polynomiale
    *   Régression Ridge
    *   Régression par Arbre de Décision
6.  **Évaluation des Modèles**: Comparaison des performances des modèles à l'aide de métriques telles que le Mean Squared Error (MSE) et le coefficient R-squared, ainsi que des visualisations des prédictions par rapport aux valeurs réelles.
7.  **Conclusion et Recommandations**: Synthèse des résultats et suggestions pour d'éventuelles améliorations futures.

## Chargement et Préparation des Données
### Chargement des Données

Cette étape consiste à charger le jeu de données `ai_job_market_insights.csv` dans un DataFrame Pandas. Nous affichons ensuite les premières lignes du DataFrame pour avoir un aperçu des données et de leur structure.
### Encodage des Variables Catégorielles

Pour préparer les données pour les modèles de machine learning, nous convertissons toutes les variables catégorielles (de type 'object') en représentations numériques. Nous utilisons `LabelEncoder` de `sklearn.preprocessing` pour attribuer un identifiant unique à chaque catégorie dans une colonne donnée. Ceci est appliqué à des colonnes comme `Job_Title`, `Industry`, `Company_Size`, etc., afin que les modèles puissent les interpréter comme des caractéristiques numériques.

## Préparation des Données pour la Modélisation
### Préparation des Données pour la Modélisation

Pour préparer nos données à la modélisation, nous devons les diviser en ensembles d'entraînement et de test. Cette étape est cruciale pour évaluer la performance du modèle sur des données non vues et éviter le surapprentissage.

*   **Ensemble d'entraînement (80%)** : Utilisé pour entraîner le modèle. Le modèle apprendra les relations entre les caractéristiques et la variable cible à partir de cet ensemble.
*   **Ensemble de test (20%)** : Utilisé pour évaluer la performance du modèle après son entraînement. Ces données n'ont jamais été vues par le modèle pendant l'entraînement.

Nous utiliserons la fonction `train_test_split` de `sklearn.model_selection` pour effectuer cette division.

*   **Variable Indépendante (X)** : Pour l'analyse initiale, nous allons sélectionner `'Job_Title'` comme notre seule variable indépendante. `X` sera donc un DataFrame contenant uniquement cette colonne.
*   **Variable Dépendante (y)** : La variable cible que nous voulons prédire est `'Salary_USD'`. `y` sera donc une Series contenant les salaires.

Le paramètre `random_state` est défini pour assurer la reproductibilité de la division des données. Cela signifie que chaque fois que le code est exécuté avec le même `random_state`, la division sera identique.

## Comparaison Finale des Modèles et Conclusion Générale
print(performance_df)
                      Model           MSE  R-squared
0         Linear Regression  4.903960e+08   0.051901
1     Polynomial Regression  4.931466e+08   0.046583
2          Ridge Regression  4.904278e+08   0.051839
3  Decision Tree Regression  8.689208e+08  -0.679914

### Analyse Détaillée et Recommandations

Le tableau comparatif ci-dessus présente les métriques d'évaluation (MSE et R-carré) pour les quatre modèles de régression que nous avons entraînés : Régression Linéaire Simple, Régression Polynomiale, Régression Ridge et Régression par Arbre de Décision.

**Observations Clés :**

1.  **Performances Générales Très Faibles** :
    *   Les modèles de régression linéaire (simple, polynomiale, Ridge) affichent des valeurs de **R-carré extrêmement faibles, autour de 0.05**. Cela signifie que seulement **environ 5% de la variance du `Salary_USD` est expliquée** par les caractéristiques que nous avons utilisées dans ces modèles. C'est un indicateur clair que ces modèles ne sont pas du tout performants pour prédire les salaires avec ce jeu de données et ces caractéristiques.
    *   Le **MSE** pour ces trois modèles est très élevé (autour de 4.90e+08), confirmant que les prédictions sont très éloignées des valeurs réelles.

2.  **Performance de la Régression par Arbre de Décision** :
    *   Le modèle de Régression par Arbre de Décision a montré une performance encore pire, avec un **R-carré négatif de -0.68** et un **MSE beaucoup plus élevé (8.69e+08)**. Un R-carré négatif est un signal fort que le modèle est une très mauvaise approximation des données, même pire que de simplement prédire la moyenne de la variable cible. Cela suggère un **sous-apprentissage** sévère ou l'incapacité du modèle à trouver des schémas significatifs dans les données.

3.  **Absence de Modèle "Performant"** : Aucun des modèles testés ne peut être qualifié de "performant" dans ce contexte. Les performances sont uniformément basses, voire très négatives pour l'arbre de décision. Les différences marginales entre la régression linéaire simple, polynomiale et Ridge ne sont pas significatives étant donné la faiblesse globale des R-carrés.

**Conclusion Générale :**

L'analyse prédictive révèle que les caractéristiques actuelles du jeu de données ne sont pas suffisamment informatives pour prédire efficacement le `Salary_USD` en utilisant des modèles de régression de base ou un arbre de décision simple. Les modèles ne parviennent pas à capturer la complexité des facteurs influençant le salaire.

**Recommandations pour les Étapes Futures :**

Pour améliorer la capacité prédictive, les étapes suivantes sont cruciales :

1.  **Ingénierie des Caractéristiques (Feature Engineering)** : C'est la priorité absolue. Les caractéristiques actuelles, bien qu'encodées, manquent probablement de puissance prédictive. Il faudrait :
    *   **Réexaminer les données brutes** : Comprendre le sens des valeurs encodées pour chaque colonne (par exemple, quelle 'Job_Title' correspond à l'encodage 0, 1, etc.) peut révéler des insights.
    *   **Créer de nouvelles caractéristiques** : Par exemple, des interactions entre 'Industry' et 'Job_Title', ou des transformations non linéaires.
    *   **Utiliser des encodages plus sophistiqués** : Pour les variables catégorielles, des techniques comme l'encodage One-Hot (si le nombre de catégories est gérable) ou des encodages basés sur des statistiques de la cible (`Target Encoding`) pourraient être plus efficaces que `LabelEncoder` qui impose un ordre arbitraire.
2.  **Collecte de Données Supplémentaires** : Le R-carré très faible suggère que des informations cruciales pour la détermination du salaire sont absentes. Des variables telles que :
    *   L'expérience professionnelle (nombre d'années)
    *   Le niveau d'éducation (diplômes, spécialisations)
    *   Les compétences spécifiques (technologies maîtrisées)
    *   La taille réelle et la réputation de l'entreprise (au lieu d'un encodage arbitraire)
    *   La performance individuelle, les responsabilités du poste
    *   Le type de contrat (CDI, CDD, freelance) pourraient être essentielles.
3.  **Exploration de Modèles Plus Robustes et Avancés** : Étant donné l'échec des modèles simples, il est impératif d'essayer des algorithmes capables de gérer des relations complexes et non linéaires :
    *   **Méthodes d'Ensemble** : Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost) sont souvent très efficaces pour les problèmes de régression.
    *   **Réseaux de Neurones** : Pour des relations très complexes, des architectures de réseaux de neurones pourraient être envisagées.
4.  **Optimisation et Validation Approfondies** : Si un modèle plus avancé montre un potentiel, une optimisation rigoureuse des hyperparamètres via des techniques comme la validation croisée et la recherche par grille/aléatoire (`GridSearchCV`, `RandomizedSearchCV`) sera nécessaire.

## Modélisation de Régression - Linéaire Simple

### Commentaires sur la Régression Linéaire Simple

Les résultats obtenus avec le modèle de régression linéaire simple sont les suivants :

*   **Mean Squared Error (MSE)** : 516 210 903.33
*   **R-squared** : 0.00

**Analyse :**

1.  **R-carré (R-squared)** : La valeur du R-carré est de 0.00. Un R-carré proche de zéro indique que le modèle linéaire simple n'explique pratiquement aucune de la variance de la variable cible (`Salary_USD`). En d'autres termes, les caractéristiques utilisées dans ce modèle (uniquement `Job_Title` dans ce cas) ne sont pas du tout efficaces pour prédire le salaire. Le modèle est incapable de capturer les relations sous-jacentes dans les données.

2.  **Mean Squared Error (MSE)** : Le MSE est de 516 210 903.33. C'est une valeur très élevée, ce qui signifie que la différence quadratique moyenne entre les salaires réels et les salaires prédits est très importante. Un MSE élevé est cohérent avec un R-carré faible et confirme que le modèle ne fournit pas de prédictions précises.

3.  **Visualisation des Prédictions vs. Valeurs Réelles** : Le nuage de points généré (`Linear Regression: Actual vs. Predicted Salaries`) montre une dispersion très large et anarchique des points par rapport à la ligne de prédiction parfaite (la ligne rouge en pointillés). Idéalement, les points devraient se regrouper étroitement autour de cette ligne diagonale pour indiquer une bonne performance du modèle. Cependant, ici, les points sont éparpillés, ne formant pas de tendance claire le long de la ligne, ce qui visuellement confirme la très faible capacité prédictive du modèle.

**Conclusion :**

Le modèle de régression linéaire simple est inefficace pour prédire le `Salary_USD` sur cet ensemble de données en utilisant uniquement le `Job_Title` comme prédicteur. Les métriques (R-carré et MSE) et la visualisation confirment que le modèle ne parvient pas à établir une relation linéaire significative entre le titre du poste et le salaire.

## Modélisation de Régression - Polynomiale

### Commentaires sur la Régression Polynomiale

Les résultats obtenus avec le modèle de régression polynomiale (degré 2) sont les suivants :

*   **Mean Squared Error (MSE)** : 493 146 641.66
*   **R-squared** : 0.0465

**Analyse :**

1.  **R-carré Très Faible :** Le R-carré de 0.0465 indique que le modèle de régression polynomiale n'explique qu'environ 4.65% de la variance des salaires (`Salary_USD`) dans le jeu de données. Cette valeur est extrêmement basse et suggère que les caractéristiques, même avec l'ajout de termes polynomiaux de degré 2, ne sont pas de bons prédicteurs des salaires. Un R-carré proche de zéro signifie que le modèle n'est pas beaucoup plus efficace que de prédire la moyenne des salaires pour toutes les observations.

2.  **MSE Élevé :** Le Mean Squared Error (MSE) est très élevé (environ 493 millions). Cela confirme que la différence quadratique moyenne entre les salaires réels et les salaires prédits par le modèle est très importante. Un MSE élevé est cohérent avec un R-carré faible et indique que les prédictions du modèle sont globalement imprécises.

3.  **Visualisation des Prédictions :** La visualisation des prédictions versus les valeurs réelles (le scatter plot généré précédemment) montrerait que les points sont très dispersés autour de la ligne de prédiction parfaite (la ligne rouge en pointillés). Il n'y aurait pas de regroupement clair des points le long de cette ligne diagonale, ce qui visuellement renforce la conclusion d'une performance médiocre du modèle.

**Conclusion sur l'Efficacité :**

L'ajout de termes polynomiaux de degré 2 à la régression n'a pas significativement amélioré la capacité de prédiction par rapport à la régression linéaire simple. En fait, le R-carré de la régression polynomiale (0.0465) est même très légèrement inférieur à celui de la régression linéaire simple (0.0519), et le MSE est légèrement supérieur. Cela suggère que les relations entre la variable `'Job_Title'` et le `Salary_USD` ne sont pas mieux capturées par une relation quadratique. Le modèle polynomial n'apporte donc pas d'amélioration notable et reste inefficace pour prédire les salaires avec cette configuration.

## Modélisation de Régression - Ridge

### Commentaires sur la Régression Ridge

Les résultats obtenus avec le modèle de régression Ridge sont les suivants :

*   **Mean Squared Error (MSE)** : 490 427 816.28
*   **R-squared** : 0.05

**Analyse :**

1.  **R-carré Très Faible :** Un R-carré de 0.05 signifie que seulement environ 5% de la variance du `Salary_USD` est expliquée par les variables indépendantes incluses dans le modèle. Cela indique que le modèle de régression Ridge, malgré l'ajout de la régularisation L2, n'est pas très performant pour prédire les salaires avec les caractéristiques actuelles.

2.  **MSE Élevé :** Le MSE est très élevé (plus de 490 millions), ce qui confirme que les prédictions du modèle sont, en moyenne, très éloignées des valeurs réelles. C'est cohérent avec la faible valeur du R-carré et souligne la difficulté du modèle à capturer la relation entre les caractéristiques et le salaire.

3.  **Visualisation :** La visualisation des prédictions (`Ridge Regression: Actual vs. Predicted Salaries`) montre un nuage de points très dispersé autour de la ligne de prédiction parfaite (la ligne rouge en pointillés). Les points ne se regroupent pas étroitement autour de cette ligne, ce qui indique que les prédictions du modèle ne suivent pas de manière fiable les salaires réels. La régularisation n'a pas conduit à une amélioration notable de la distribution des points par rapport aux modèles linéaires précédents.

**Conclusion :**

Le modèle de régression Ridge, dans cette configuration et avec les caractéristiques disponibles, n'est pas efficace pour prédire le `Salary_USD`. Ses performances sont similaires à celles de la régression linéaire simple et de la régression polynomiale de degré 2, ce qui suggère que l'ajout d'une pénalité L2 n'a pas permis de résoudre les problèmes fondamentaux liés au manque de pouvoir explicatif des caractéristiques ou à la complexité de la relation à modéliser. Pour améliorer les prédictions, il serait nécessaire d'explorer une ingénierie des caractéristiques plus poussée, d'intégrer des données supplémentaires plus riches, ou d'utiliser des modèles de régression plus complexes et non linéaires.

## Modélisation de Régression - Arbre de Décision

### Commentaires sur la Régression par Arbre de Décision

Les résultats obtenus avec le modèle de régression par arbre de décision sont les suivants :

*   **Mean Squared Error (MSE)** : 868 920 848.19
*   **R-squared** : -0.68

**Analyse :**

1.  **R-carré Négatif :** Un R-carré négatif est un indicateur très fort que le modèle est une très mauvaise approximation des données. Il signifie que le modèle de régression est pire que le simple fait de prédire la moyenne de la variable dépendante (`Salary_USD`) pour toutes les observations. Dans ce cas (-0.68), le modèle par arbre de décision ne parvient absolument pas à capturer la variance des salaires et fait des prédictions très éloignées de la réalité.

2.  **MSE Élevé :** Le MSE est extrêmement élevé, ce qui confirme que la différence quadratique moyenne entre les valeurs réelles et prédites est très importante. Cela est cohérent avec le R-carré négatif et indique une faible précision des prédictions.

3.  **Visualisation :** Le nuage de points des salaires réels versus prédits, comparé à la ligne de prédiction parfaite (rouge en pointillés), montrerait une dispersion très large et anarchique des points. On ne s'attendrait pas à voir les points se regrouper le long de la ligne diagonale, confirmant visuellement la performance médiocre du modèle.

**Conclusion :**

Le modèle de régression par arbre de décision, dans sa configuration par défaut ou actuelle, n'est pas du tout adapté pour prédire les salaires avec cet ensemble de données. Sa performance est même inférieure à celle d'un modèle basique qui prédirait simplement la moyenne des salaires. Cela suggère plusieurs points :

*   **Complexité du problème :** La relation entre les caractéristiques disponibles et le salaire est probablement très complexe et non linéaire, et l'arbre de décision seul ne parvient pas à la modéliser.
*   **Absence de surapprentissage :** Bien que les arbres de décision puissent facilement surapprendre, un R-carré négatif ici indique qu'il y a un fort sous-apprentissage ou que le modèle n'a pas réussi à trouver des schémas utiles.
*   **Qualité/Quantité des données :** Comme observé avec les modèles linéaires, les caractéristiques encodées pourraient ne pas contenir suffisamment d'informations prédictives pour le salaire.

**Prochaines étapes possibles :**

Pour les arbres de décision, il serait essentiel d'explorer l'optimisation des hyperparamètres (comme la profondeur maximale de l'arbre, le nombre minimum d'échantillons pour diviser un nœud, etc.) pour tenter d'améliorer sa performance. Cependant, étant donné le R-carré fortement négatif, il est plus probable que ce type de modèle seul ne soit pas suffisant et que des approches plus robustes comme les méthodes d'ensemble (Random Forest, Gradient Boosting) soient nécessaires, potentiellement combinées à une meilleure ingénierie des caractéristiques.


