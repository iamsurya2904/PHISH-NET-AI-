import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the phishing dataset
dataset_path = 'phishing.csv'  # Ensure this file is in the same directory
try:
    data = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError("Dataset 'phishing.csv' not found. Ensure it is in the same directory.")

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last
y = data.iloc[:, -1]   # Last column contains labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier()
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
model_path = 'model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_path}")
