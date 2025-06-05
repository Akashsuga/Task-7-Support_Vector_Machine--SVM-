# Task-7-Support_Vector_Machine--SVM-
# Breast Cancer Classification using Support Vector Machine (SVM)

## Project Overview

This project uses Support Vector Machines (SVM) to classify tumors in the Breast Cancer dataset as benign or malignant. It compares linear and RBF kernels, utilizes PCA for 2D visual interpretation, applies hyperparameter tuning, and evaluates performance using cross-validation.

---

## Repository Structure

- **breast-cancer.csv** — Original dataset
- **task_7_svm.py** — Python script for Support Vector Machine (SVM)
- **task 7.pdf** — Given task
- **Visual Outputs** — Folder containing all visual plots
- **README.md** — This documentation

---

## Dataset

- Filename: breast-cancer.csv
- Dataset Download Link : https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset 
- Target column: diagnosis (M = Malignant, B = Benign)
- Features: Numerical measurements from diagnostic tests

---

## Tools and Libraries Used

- pandas, numpy
- matplotlib, seaborn
- scikit-learn (SVC, PCA, GridSearchCV, classification_report, confusion_matrix)

---

## Steps Performed

1. Data cleaning (drop non-numeric columns, handle missing values)
2. Label encoding (M → 1, B → 0)
3. Train-test split (80% train, 20% test)
4. Feature scaling with StandardScaler
5. Train SVM models:
   - Linear kernel
   - RBF kernel
6. Project features to 2D using PCA
7. Visualize decision boundaries
8. Tune hyperparameters with GridSearchCV
9. Cross-validation (5-fold)
10. Evaluate using confusion matrix and classification report

---

## Visualizations

Below are the five key visuals generated in the analysis:

### 1. Linear SVM Decision Boundary (PCA Projection)
Shows separation in 2D feature space using linear kernel.
![Linear SVM Decision Boundary (PCA)](https://github.com/user-attachments/assets/c9a2d325-00a4-431e-a6c2-bc6b8cab1338)

### 2. RBF SVM Decision Boundary (PCA Projection)
Shows nonlinear decision boundary using RBF kernel.
![RBF SVM Decision Boundary (PCA)](https://github.com/user-attachments/assets/6b941c99-6d46-404e-bb0a-7e74ac707b21)

### 3. Confusion Matrix
Displays true vs predicted classifications.
![Confusion Matrix](https://github.com/user-attachments/assets/d1786546-ca81-4c92-8e4f-c094ef04e7d7)

### 4. Classification Report Heatmap
Visualizes precision, recall, F1-score for each class.
![Classification Report](https://github.com/user-attachments/assets/9c521f8c-c7c7-46cf-be9b-89c1fd112d53)

### 5. Cross-Validation Accuracy per Fold
Bar plot of model accuracy across cross-validation folds.![CV Accuracy per fold](https://github.com/user-attachments/assets/f60ddb8a-c6d9-4659-8fd6-d4091571c180)


---

## Insights and Patterns

- PCA shows two partially overlapping but distinguishable clusters.
- RBF kernel captures non-linear relationships better than linear kernel.
- Consistent performance across cross-validation folds indicates good generalization.

---

## Anomalies and Observations

- Minor overlap in features leads to some misclassification.
- Slight class imbalance is present (more benign cases).

---

## Conclusion

- SVM with RBF kernel is highly effective in classifying tumors.
- Visual analysis using PCA helps explain decision regions.
- GridSearchCV helps optimize hyperparameters efficiently.

---

## Author

- **AKASH M** 
- Internship Task 4 - Binary Classification using SVM  
- Tools: Python, scikit-learn, matplotlib, seaborn

