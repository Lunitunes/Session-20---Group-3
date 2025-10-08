import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

training = pd.read_csv("processed_data/basic/trainingDataset.csv")
validation = pd.read_csv("processed_data/basic/validationDataset.csv")

X_train = training.drop(columns=['encodedCategory'])
y_train = training['encodedCategory']

X_test = validation.drop(columns=['encodedCategory'])
y_test = validation['encodedCategory']

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, dt_preds)
precision = precision_score(y_test, dt_preds, average='weighted')
recall = recall_score(y_test, dt_preds, average='weighted')
f1 = f1_score(y_test, dt_preds, average='weighted')

print(f"\nModel Performance:")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

print("\nDetailed Classification Report:\n")
print(classification_report(y_test, dt_preds))
