# Install required packages
import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = ["pandas", "numpy", "scikit-learn", "statsmodels", "matplotlib", "seaborn"]

# Install each package
for package in packages:
    install(package)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

## Explore Data ##
# Specify the file path
file_path = "/Users/zeyadosama/Desktop/CodeAlpha/Task1/train.csv"
Ship = pd.read_csv(file_path)
print(Ship.info())
print(Ship.describe())

## Data Preparation ##

# Remove insignificant Variables
Ship.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Handle Missing values at the 'Age' variable with mean imputation.
Ship['Age'].fillna(Ship['Age'].mean(), inplace=True)
print(Ship.describe())

# Transform categorical factors
Ship['Sex'] = Ship['Sex'].apply(lambda x: 1 if x == 'male' else 0)
Ship['Embarked'] = Ship['Embarked'].apply(lambda x: 1 if x == 'C' else (2 if x == 'Q' else 3))
print(Ship.info())
print(Ship.describe())

# Calculate the Baseline Accuracy
baseline_accuracy = Ship['Survived'].value_counts().max() / Ship['Survived'].value_counts().sum()
print(f"Baseline Accuracy: {baseline_accuracy}")

# Split the data for training and testing
X = Ship.drop('Survived', axis=1)
y = Ship['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=144)

## Logistic Regression ##
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']], y_train)

# Logistic model after removing insignificant variables (Parch, Embarked)
logistic_model_2 = LogisticRegression(max_iter=200)
logistic_model_2.fit(X_train[['Pclass', 'Sex', 'Age', 'SibSp']], y_train)

# Make predictions on testing set
predictTest = logistic_model_2.predict_proba(X_test[['Pclass', 'Sex', 'Age', 'SibSp']])[:, 1]

# Analyze predictions
print(pd.Series(predictTest).describe())
print(pd.DataFrame({'Survived': y_test, 'PredictTest': predictTest}).groupby('Survived')['PredictTest'].mean())

# Create the ROCR curve
fpr, tpr, thresholds = roc_curve(y_test, predictTest)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Add colors and threshold labels
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
for i, threshold in enumerate(thresholds):
    if i % 10 == 0:
        plt.text(fpr[i], tpr[i], f'{threshold:.1f}', fontsize=9, ha='right')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic with Thresholds')
plt.show()

# Confusion matrix for threshold of 0.3
threshold = 0.3
conf_matrix = confusion_matrix(y_test, predictTest > threshold)

# Extract values from the confusion matrix
true_negatives_L = conf_matrix[0, 0]
false_positives_L = conf_matrix[0, 1]
false_negatives_L = conf_matrix[1, 0]
true_positives_L = conf_matrix[1, 1]

# Calculate sensitivity, specificity, and accuracy
sensitivity_Log = true_positives_L / (true_positives_L + false_negatives_L)
specificity_Log = true_negatives_L / (true_negatives_L + false_positives_L)
accuracy_Log = (true_positives_L + true_negatives_L) / conf_matrix.sum()

# Print the results
print(f"Sensitivity: {sensitivity_Log}")
print(f"Specificity: {specificity_Log}")
print(f"Accuracy: {accuracy_Log}")

## CART model ##
tree1 = DecisionTreeClassifier()
tree1.fit(X_train, y_train)

# Plot the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(tree1, filled=True)
plt.show()

# There are 4 splits
# The accuracy of the model
CARTpred = tree1.predict(X_test)

conf_matrix_cart = confusion_matrix(y_test, CARTpred)

# Extract values from the confusion matrix
true_negatives_C = conf_matrix_cart[0, 0]
false_positives_C = conf_matrix_cart[0, 1]
false_negatives_C = conf_matrix_cart[1, 0]
true_positives_C = conf_matrix_cart[1, 1]

# Calculate sensitivity, specificity, and accuracy for the CART model
sensitivity_C = true_positives_C / (true_positives_C + false_negatives_C)
specificity_C = true_negatives_C / (true_negatives_C + false_positives_C)
accuracy_C = (true_positives_C + true_negatives_C) / conf_matrix_cart.sum()

# Print the results
print(f"Sensitivity_C: {sensitivity_C}")
print(f"Specificity_C: {specificity_C}")
print(f"Accuracy_C: {accuracy_C}")

## Random Forest Model ##
RanFor = RandomForestClassifier(random_state=1)
RanFor.fit(X_train, y_train)
variable_importance = RanFor.feature_importances_
print(variable_importance)
Forpred = RanFor.predict(X_test)

# Confusion matrix for RF
conf_matrix_RF = confusion_matrix(y_test, Forpred)

# Sensitivity, Specificity, and Accuracy for RF
sensitivity_RF = conf_matrix_RF[1, 1] / conf_matrix_RF[1].sum()
specificity_RF = conf_matrix_RF[0, 0] / conf_matrix_RF[0].sum()
accuracy_RF = conf_matrix_RF.diagonal().sum() / conf_matrix_RF.sum()

# Print the results for RF
print(f"Sensitivity_RF: {sensitivity_RF}")
print(f"Specificity_RF: {specificity_RF}")
print(f"Accuracy_RF: {accuracy_RF}")