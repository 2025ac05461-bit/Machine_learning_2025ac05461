
a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a breast mass is malignant or benign using characteristics computed from digitized images of breast mass fine needle aspirates.

The project implements and evaluates five classification algorithms and compares their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

b. Dataset Description

Breast Cancer Wisconsin (Diagnostic) Dataset

The Breast Cancer Wisconsin (Diagnostic) Dataset was originally hosted by the UCI Machine Learning Repository and is also available as a built-in dataset in scikit-learn.

- Instances: 569
- Features: 30 numeric features
- Target: Binary classification
- 0 = Malignant
- 1 = Benign
- Class balance: 212 malignant / 357 benign

The features describe characteristics of cell nuclei computed from digitized images of breast mass fine needle aspirates, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

An 80/20 stratified train-test split was used with random state 42. StandardScaler was used for Logistic Regression, kNN, and Gaussian Naive Bayes; tree-based models were trained without scaling.

c. GitHub Repository Link

https://github.com/2025ac05461-bit/Machine_learning_2025ac05461

d. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)

Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |


Logistic Regression - 
Best-performing model overall. It achieved the highest Accuracy, AUC, Precision, Recall, F1, and MCC. 
Decision Tree -
 Weakest-performing model overall, with the lowest Accuracy, AUC, Recall, F1, and MCC.
kNN - 
Strong performer with high Accuracy, Recall, F1, and MCC. Its distance-based nature makes feature standardization important. 
Naive Bayes - 
Lower Accuracy than kNN and Random Forest, but a strong AUC of 0.9868 indicates good class-separation performance.
Random Forest (Ensemble)  - 
Strong ensemble model with 94.74% Accuracy and 0.9937 AUC, but it did not outperform Logistic Regression on the other reported metrics.
Overall Best performing model - Logistic Regression as it achieved the highest score on all six reported evaluation metrics.

 Repository Structure


classification-model-comparison/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.py
    ├── logistic_regression.pkl
    ├── decision_tree.py
    ├── decision_tree.pkl
    ├── knn.py
    ├── knn.pkl
    ├── naive_bayes.py
    ├── naive_bayes.pkl
    ├── random_forest.py
    └── random_forest.pkl
