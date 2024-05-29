import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import joblib

# Load dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Drop unnecessary columns
dataset = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

dataset.info()

# One-hot encoding for categorical variables
dataset = pd.get_dummies(data=dataset, drop_first=True)

# Split dataset into features (X) and target variable (y)
X = dataset.drop(columns='Exited')
y = dataset['Exited']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
logistic_clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred_logistic = logistic_clf.predict(X_test)

# Evaluate Logistic Regression model
logistic_acc = accuracy_score(y_test, y_pred_logistic)
logistic_f1 = f1_score(y_test, y_pred_logistic)
logistic_prec = precision_score(y_test, y_pred_logistic)
logistic_rec = recall_score(y_test, y_pred_logistic)

# Train Random Forest model
random_forest_clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_pred_rf = random_forest_clf.predict(X_test)

# Evaluate Random Forest model
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)

# Confusion matrices
logistic_cm = confusion_matrix(y_test, y_pred_logistic)
rf_cm = confusion_matrix(y_test, y_pred_rf)

# Save Random Forest model to file
joblib.dump(random_forest_clf, 'random_forest_model.pkl')

# Results DataFrame
results = pd.DataFrame([
    ['Logistic Regression', logistic_acc, logistic_f1, logistic_prec, logistic_rec],
    ['Random Forest Classifier', rf_acc, rf_f1, rf_prec, rf_rec]],
    columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])

# Print results and confusion matrices
print("Results:")
print(results)
print("\nConfusion Matrix for Logistic Regression:")
print(logistic_cm)
print("\nConfusion Matrix for Random Forest Classifier:")
print(rf_cm)

# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(logistic_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot confusion matrix for Random Forest Classifier
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

