import numpy as np
import pandas as pd
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
 
#Load data
training = pd.read_csv("processed_data/nb15/trainingDataset.csv")
validation = pd.read_csv("processed_data/nb15/validationDataset.csv")
 
X_train = training.drop(columns=['encodedCategory'])
y_train = training['encodedCategory']
 
X_test  = validation.drop(columns=['encodedCategory'])
y_test  = validation['encodedCategory']
 
#Logistic Regression (multiclass, class-weighted)
lg_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    multi_class="auto",
    solver="lbfgs",
    random_state=42
)
 
lg_model.fit(X_train, y_train)
lg_preds = lg_model.predict(X_test)
 
#Metrics
acc   = accuracy_score(y_test, lg_preds)
precw = precision_score(y_test, lg_preds, average='weighted', zero_division=0)
recw  = recall_score(y_test, lg_preds, average='weighted', zero_division=0)
f1w   = f1_score(y_test, lg_preds, average='weighted', zero_division=0)
f1m   = f1_score(y_test, lg_preds, average='macro', zero_division=0)
 
print("\nModel Performance (Logistic Regression, class-weighted):")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision (weighted): {precw:.3f}")
print(f"Recall    (weighted): {recw:.3f}")
print(f"F1        (weighted): {f1w:.3f}")
print(f"F1        (macro)   : {f1m:.3f}")
 
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, lg_preds, digits=3, zero_division=0))
 
print("\nConfusion Matrix (rows=true, cols=pred):\n")
print(confusion_matrix(y_test, lg_preds))
 
