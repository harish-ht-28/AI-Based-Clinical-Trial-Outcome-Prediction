import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
num_samples = 500

age = np.random.randint(25, 80, num_samples)
drug_dosage = np.random.uniform(10, 100, num_samples)
biomarker_levels = np.random.uniform(0.1, 5.0, num_samples)

# Outcome (1 = Success, 0 = Failure)
outcome = (0.3 * age + 0.5 * drug_dosage + 0.2 * biomarker_levels + np.random.normal(0, 10, num_samples)) > 50
outcome = outcome.astype(int)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Drug Dosage': drug_dosage,
    'Biomarker Levels': biomarker_levels,
    'Outcome': outcome
})

X = data[['Age', 'Drug Dosage', 'Biomarker Levels']]
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output results
print("Model Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Visualization
plt.figure(figsize=(6, 4))
plt.bar(['Failure', 'Success'], [np.sum(y_test == 0), np.sum(y_test == 1)], color=['red', 'green'])
plt.title('Actual Clinical Trial Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()
