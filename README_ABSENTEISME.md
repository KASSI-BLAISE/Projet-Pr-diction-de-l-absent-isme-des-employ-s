# 👔 Prédiction de l'absentéisme des employés

Projet de classification pour anticiper le risque d'absentéisme prolongé, à partir de données RH tabulaires, avec une application desktop d'aide à la décision.

---

## 📌 Cahier des charges

- **Type ML** : Classification
- **Données** : Données RH tabulaires (740 employés, dataset [Absenteeism at Work](https://archive.ics.uci.edu/dataset/445/absenteeism+at+work), UCI Machine Learning Repository)
- **Techniques demandées** :
  - Ingénierie temporelle (jours, saisons)
  - Encodage avancé des catégories
  - Sélection de variables
  - Modèles : Logistic Regression, RandomForest, XGBoost
  - Fusion de modèles : Stacking
- **Application** : Application RH d'aide à la décision

## 🎯 Démarche

### Définition de la cible
La variable brute `Absenteeism time in hours` est continue (0 à 120h), incompatible avec une tâche de classification. Elle a été binarisée à un seuil de **4 heures** : *absence courte (<4h)* vs *absence longue (≥4h)*, un seuil qui donne une répartition quasi équilibrée (45,8% / 54,2%) et une distinction actionnable pour un service RH.

### Ingénierie temporelle
- **Encodage cyclique (sin/cos)** du mois et du jour de la semaine, pour que le modèle comprenne la continuité circulaire du temps (ex. décembre et janvier sont proches, pas opposés)
- Variables dérivées : début/fin de semaine, saison estivale

### Encodage avancé des catégories
La variable `Reason for absence` (28 codes médicaux CID) a été traitée par **target encoding avec lissage**, plutôt qu'un one-hot qui aurait dilué le signal sur 28 colonnes creuses. Le lissage évite le surapprentissage sur les catégories peu représentées (calcul strictement réalisé sur le train, pour écarter tout risque de fuite de données).

### Sélection de variables
Feature importance via RandomForest : 15 variables sur 24 retenues pour 90% de l'importance cumulée. `Reason_encoded` domine largement (33,6% de l'importance à lui seul), suivi de la charge de travail, l'atteinte des objectifs, et les variables temporelles cycliques — validant leur utilité.

### Modèles individuels

| Modèle | Accuracy | F1 (classe "longue") |
|---|---|---|
| Logistic Regression | 0,777 | 0,76 |
| **RandomForest** | **0,791** | **0,77** |
| XGBoost | 0,784 | 0,77 |

### Fusion de modèles (Stacking)

| Modèle | Accuracy |
|---|---|
| Logistic Regression | 0,777 |
| **RandomForest** | **0,791** |
| XGBoost | 0,784 |
| Stacking (méta-modèle : Logistic Regression) | 0,770 |

**Conclusion méthodologique : le Stacking ne bat aucun des trois modèles individuels**, et se classe même dernier. Sur un dataset de taille modeste (740 lignes), la validation croisée interne du Stacking (5 folds) réduit encore les données disponibles pour chaque modèle de base, ce qui peut dégrader la qualité de la fusion. C'est le deuxième projet de ce portfolio (après la reconnaissance de chiffres MNIST) où une technique de fusion de modèles, bien que rigoureusement testée, n'apporte pas de gain par rapport au meilleur modèle individuel — un résultat assumé et documenté plutôt que forcé.

**Modèle retenu : RandomForest seul (79,1% d'accuracy).**

---

## 🖥️ Application desktop (Tkinter)

Interface de saisie du profil d'un employé (motif d'absence, données temporelles, charge de travail, données personnelles), avec :
- Prédiction du risque d'absence longue vs courte
- Détail des probabilités par classe
- Calcul automatique de l'IMC
- Mention explicite : outil d'aide à la décision, à utiliser en complément du jugement RH, jamais en remplacement

## 🛠️ Stack technique

- **Python** — pandas, numpy
- **scikit-learn** — LogisticRegression, RandomForest, StackingClassifier, StandardScaler
- **XGBoost**
- **Tkinter** — application desktop

## 🚀 Installation et lancement

```bash
git clone https://github.com/KASSI-BLAISE/<nom-du-repo>.git
cd <nom-du-repo>
pip install -r requirements.txt
python app_desktop.py
```

## 📁 Structure du dépôt

```
prediction-absenteisme/
├── app_desktop.py
├── ProjetAbsenteisme.ipynb
├── model_absenteisme_rf.pkl
├── reason_encoding_map.pkl
├── moyenne_globale_reason.pkl
├── variables_selectionnees.pkl
├── requirements.txt
└── README.md
```

## 📦 Données

Dataset : [Absenteeism at Work](https://archive.ics.uci.edu/dataset/445/absenteeism+at+work) (UCI Machine Learning Repository) — 740 enregistrements, 21 variables, données d'une entreprise de courrier brésilienne (juillet 2007 à juillet 2010).

## 🔭 Pistes d'amélioration

- Explorer un seuil de classification différent (ex. 8h, séparant mieux les absences "journée complète")
- Enrichir l'encodage des 28 motifs d'absence avec leur libellé médical complet (actuellement simplifié dans l'application)
- Tester un Stacking avec un méta-modèle non-linéaire, ou une pondération des modèles de base selon leur performance individuelle

---

*Projet réalisé dans le cadre d'une transition professionnelle vers la data science. Illustre une démarche de validation empirique systématique des techniques de fusion de modèles, plutôt que leur application par défaut.*
