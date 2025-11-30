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

## Analyse Exploratoire des Données (EDA)

L'Analyse Exploratoire des Données (EDA) est une étape cruciale pour comprendre la structure, les modèles, les valeurs aberrantes et les relations au sein de notre jeu de données. L'objectif principal de cette section est d'explorer la distribution de notre variable cible, `Salary_USD`, et d'identifier les relations potentielles entre le salaire et les différentes caractéristiques indépendantes.

Nous allons utiliser plusieurs visualisations pour atteindre cet objectif :
1.  Un histogramme pour visualiser la distribution globale de `Salary_USD`, nous permettant de comprendre sa forme, sa tendance centrale et sa dispersion.
2.  Des box plots pour examiner la distribution de `Salary_USD` en fonction de chaque variable catégorielle encodée (`Job_Title`, `Industry`, `Company_Size`, `Location`, `AI_Adoption_Level`, `Automation_Risk`, `Required_Skills`, `Remote_Friendly`, et `Job_Growth_Projection`). Ces visualisations nous aideront à identifier comment le salaire varie en fonction des différentes catégories de chaque caractéristique, et à détecter d'éventuelles différences significatives ou tendances.

### Distribution de Salary_USD

Cet histogramme illustre la distribution de la variable cible `Salary_USD`. La ligne `kde` (Kernel Density Estimate) fournit une représentation lissée de la distribution, aidant à identifier sa forme, sa tendance centrale et sa dispersion. Nous pouvons observer la plage des salaires et où se situe la majorité des salaires au sein de l'ensemble de données. Généralement, la distribution semble suivre une forme quasi-normale, légèrement asymétrique, avec la plupart des salaires regroupés autour de la moyenne, et une diminution de la fréquence à mesure que l'on s'éloigne de cette moyenne. Cela nous donne une première idée de la répartition des salaires dans le marché de l'emploi analysé.

### Analyse des Relations entre les Variables Indépendantes et Salary_USD

Ces box plots illustrent la relation entre chaque caractéristique indépendante encodée et la variable cible `Salary_USD`. Chaque box plot représente la médiane, les quartiles et les valeurs aberrantes potentielles de `Salary_USD` pour chaque catégorie de la caractéristique indépendante.

En examinant ces graphiques, nous pouvons tirer les observations suivantes :

*   **Job_Title** : Les salaires varient considérablement selon les différentes catégories de `Job_Title`. Certains titres de poste (représentés par leurs codes encodés) montrent des médianes salariales plus élevées et une plus grande dispersion, tandis que d'autres ont des salaires plus concentrés ou plus faibles. Cela suggère que le type de poste est un facteur important influençant le salaire.

*   **Industry** : La distribution des salaires diffère également d'une `Industry` à l'autre. Certaines industries peuvent offrir des salaires médians plus élevés ou une gamme de salaires plus large que d'autres. L'encodage numérique permet de comparer les tendances générales, même si les étiquettes originales ne sont pas immédiatement visibles.

*   **Company_Size** : Les box plots pour `Company_Size` indiquent si la taille de l'entreprise (petite, moyenne, grande, encodée) a un impact sur le niveau de salaire. On peut observer si les grandes entreprises, par exemple, ont tendance à payer plus, ou si la dispersion des salaires est différente.

*   **Location** : La `Location` géographique (encodée) est un facteur classique d'influence sur les salaires. Les graphiques révèlent si certaines localisations ont des salaires médians nettement supérieurs ou inférieurs, reflétant potentiellement le coût de la vie ou la demande locale pour certains postes.

*   **AI_Adoption_Level** : L'impact du `AI_Adoption_Level` sur les salaires est également visualisé. Il est intéressant de voir si un niveau d'adoption de l'IA plus élevé (encodé) correspond à des salaires supérieurs, ce qui pourrait indiquer une valorisation des compétences liées à l'IA.

*   **Automation_Risk** : Ces graphiques montrent comment les salaires se distribuent en fonction du `Automation_Risk`. On pourrait s'attendre à des salaires plus élevés pour les postes à faible risque d'automatisation, ou des différences de dispersion.

*   **Required_Skills** : La variable `Required_Skills` (encodée) peut également montrer des variations salariales. Des compétences rares ou très demandées pourraient être associées à des salaires plus élevés.

*   **Remote_Friendly** : La flexibilité du travail (`Remote_Friendly`) peut aussi influencer les salaires. Les box plots permettent d'analyser si les postes télétravaillables (encodés) ont des distributions salariales différentes de ceux qui ne le sont pas.

*   **Job_Growth_Projection** : Enfin, la `Job_Growth_Projection` (encodée) peut être liée aux salaires. Les secteurs ou postes avec une forte projection de croissance peuvent parfois offrir des salaires plus compétitifs.

En examinant l'ensemble de ces visualisations, nous pouvons identifier quelles caractéristiques semblent avoir une influence plus prononcée sur les salaires, même si les variables sont encodées. Cependant, des différences subtiles peuvent être difficiles à interpréter sans les mappings des valeurs encodées aux catégories originales.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Display the correlation matrix
print("Matrice de Corrélation:")
print(correlation_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation des Variables')
plt.show()
### Analyse des Corrélations

La carte de chaleur de la matrice de corrélation révèle les relations linéaires entre toutes les variables du jeu de données, y compris la variable cible `Salary_USD`.

**Observations Clés :**

1.  **Corrélations avec `Salary_USD` :**
    *   **Automation_Risk** : Il existe une faible corrélation négative (-0.12) avec `Salary_USD`. Cela pourrait suggérer que les postes présentant un risque d'automatisation plus élevé ont tendance à avoir des salaires légèrement inférieurs, ou vice-versa. Cependant, la force de cette corrélation est très faible, ce qui signifie que l'impact est marginal.
    *   **Industry** : On observe une faible corrélation négative (-0.10) avec `Salary_USD`. Ceci pourrait indiquer que certaines industries (après encodage numérique) paient légèrement moins que d'autres, mais là encore, l'effet est très limité.
    *   **AI_Adoption_Level** : Il y a une très faible corrélation positive (0.08) avec `Salary_USD`. Cela pourrait insinuer que les salaires sont marginalement plus élevés dans les entreprises avec un niveau d'adoption de l'IA plus important, mais la corrélation est trop faible pour tirer des conclusions significatives.
    *   **Required_Skills** : Une corrélation positive très faible (0.04) est présente, ce qui suggère une influence presque négligeable des compétences requises (après encodage) sur le salaire.
    *   Les autres variables (`Job_Title`, `Company_Size`, `Location`, `Remote_Friendly`, `Job_Growth_Projection`) montrent des corrélations avec `Salary_USD` qui sont très proches de zéro, indiquant qu'elles ont très peu ou pas de relation linéaire directe avec le salaire dans ce jeu de données tel qu'encodé.

2.  **Corrélations entre Variables Indépendantes :**
    *   La plupart des corrélations entre les variables indépendantes sont également très faibles, proches de zéro. Cela est généralement une bonne chose pour les modèles de régression linéaire, car cela signifie qu'il n'y a pas de forte multicolinéarité entre les prédicteurs.
    *   Aucune corrélation particulièrement forte (positive ou négative) n'est observable entre les différentes caractéristiques, ce qui indique que les variables indépendantes sont relativement indépendantes les unes des autres.

**Implications :**

Les faibles valeurs de corrélation observées entre les variables indépendantes et `Salary_USD` sont cohérentes avec les résultats des modèles de régression linéaire précédents (linéaire simple, polynomiale et Ridge). Un R-carré très faible pour tous ces modèles suggérait déjà que les caractéristiques actuelles n'expliquent qu'une très petite partie de la variance du salaire. La matrice de corrélation confirme que peu de ces caractéristiques, si tant est qu'il y en ait, ont une relation linéaire significative avec le salaire.

Cela renforce la conclusion que des efforts sont nécessaires en matière d'ingénierie des caractéristiques ou que des données supplémentaires/plus riches sont requises pour construire un modèle de prédiction de salaire plus performant. Il est également possible que la relation entre ces variables et le salaire ne soit pas linéaire et ne soit donc pas capturée par cette analyse de corrélation de Pearson.
## Préparation des Données pour la Modélisation

Pour préparer les données à la modélisation, nous avons suivi les étapes suivantes :

1.  **Sélection des Variables :**
    *   Nous avons identifié la variable cible (dépendante) comme `'Salary_USD'`, car c'est ce que nous souhaitons prédire. Cette variable sera représentée par `y`.
    *   Toutes les autres colonnes du DataFrame `df`, à l'exception de `'Salary_USD'`, sont considérées comme des variables indépendantes (prédicteurs). Ces variables seront représentées par `X`. Il est important de noter que toutes ces variables ont déjà été encodées numériquement lors des étapes précédentes.

2.  **Division en Ensembles d'Entraînement et de Test :**
    *   Afin d'évaluer la performance de nos modèles sur des données non vues, nous avons divisé l'ensemble de données en deux sous-ensembles : un ensemble d'entraînement (`X_train`, `y_train`) et un ensemble de test (`X_test`, `y_test`).
    *   La fonction `train_test_split` de `sklearn.model_selection` a été utilisée pour effectuer cette division.
    *   Nous avons alloué **80%** des données pour l'entraînement (`train_size=0.8` ou `test_size=0.2`) et **20%** pour le test.
    *   Un `random_state` de `42` a été défini pour garantir la reproductibilité de la division des données. Cela signifie que si le code est exécuté plusieurs fois, la même répartition des données sera obtenue, ce qui est crucial pour comparer les performances des modèles de manière cohérente.

