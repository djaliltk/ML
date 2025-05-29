Introduction
Chaque jour, des milliards de transactions par carte bancaire sont effectuées dans le monde.
Avec l’explosion des paiements en ligne via smartphones et applications, il devient essentiel de pouvoir détecter rapidement les fraudes.

Dans ce projet, j'ai utilisé scikit-learn pour construire un modèle capable de différencier les transactions frauduleuses des transactions légitimes.
Plusieurs modèles ont été testés afin de choisir celui offrant les meilleures performances.






Objectif du projet
L'objectif est clair :
Minimiser les fraudes sans pour autant bloquer trop de transactions authentiques.

Pour cela, j’ai entraîné et comparé quatre algorithmes :

Decision Tree

Random Forest

AdaBoost Classifier

Gradient Boosting







Données utilisées
Le dataset provient de Kaggle et contient les transactions bancaires de clients européens sur deux jours de septembre 2013.

Quelques caractéristiques importantes du dataset :

Time : secondes écoulées depuis la première transaction.

Amount : montant de la transaction.

Class : 0 = transaction normale, 1 = fraude.

Les autres variables (V1 à V28) sont issues d'une PCA pour protéger les données confidentielles.






Préparation des données

Avant d'entraîner les modèles, j’ai effectué :

Une normalisation des montants avec StandardScaler.

Un équilibrage des classes (fraudes rares) avec SMOTE pour éviter le biais du modèle.





Méthodes d'évaluation
Pour évaluer les performances, plusieurs métriques ont été utilisées :

Accuracy : pourcentage de bonnes prédictions.

Precision : proportion de vraies fraudes parmi les fraudes détectées.

Recall : proportion de fraudes détectées parmi toutes les fraudes existantes.

F1-Score : moyenne harmonique entre précision et rappel.

Matrice de confusion : visualisation des vrais positifs, faux positifs, vrais négatifs, et faux négatifs.







Résultats obtenus
Parmi les modèles testés :

AdaBoost Classifier a atteint le meilleur rappel (91,87%), détectant 147 fraudes sur 160.

Toutefois, il a généré plus de faux positifs (1,54% des transactions normales mal classées).

Le Random Forest a été plus conservateur :

Moins de fraudes détectées (recall à 82,50%),

Mais beaucoup moins de faux positifs (0,02%).






Conclusion
En détection de fraude, tout est question d'équilibre :
Faut-il détecter un maximum de fraudes quitte à avoir plus de faux positifs ?
Ou faut-il réduire les alertes inutiles mais risquer de laisser passer quelques fraudes ?

Le choix final du modèle dépendra des priorités du business et de l'impact des erreurs sur l'expérience client.












0-install requirements

1-add ur dataset(creditecard)

2-run ipynb

3-run save models

4-run app (upload creaditcard2023)


