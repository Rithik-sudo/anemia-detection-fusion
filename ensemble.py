import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
csv_path = "D:/Work-Related/fiverr/fusion/anemia_cleaned.csv"  # Update with the actual file path
df = pd.read_csv(csv_path)

# Separate features and labels
X = df.drop(columns=["Result"])  # Features
y = df["Result"]  # Target variable (0 = Non-Anemic, 1 = Anemic)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Initialize ML models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, kernel="rbf", random_state=42)  # Enable probability for soft voting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train models
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Save individual models
joblib.dump(rf, "random_forest.pkl")
joblib.dump(svm, "svm.pkl")
joblib.dump(gb, "gradient_boosting.pkl")

# Create an ensemble using soft voting
ensemble_model = VotingClassifier(estimators=[("rf", rf), ("svm", svm), ("gb", gb)], voting="soft")

# Train the ensemble
ensemble_model.fit(X_train, y_train)

# Save the ensemble model
joblib.dump(ensemble_model, "ensemble_model.pkl")

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Anemic", "Anemic"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n🔹 Model: {model_name}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("📊 Classification Report:\n", report)
    print("🛑 Confusion Matrix:\n", cm)

# Evaluate each model
evaluate_model(rf, X_test, y_test, "Random Forest")
evaluate_model(svm, X_test, y_test, "SVM")
evaluate_model(gb, X_test, y_test, "Gradient Boosting")
evaluate_model(ensemble_model, X_test, y_test, "Ensemble Model")

print("\n✅ All models trained, evaluated, and saved successfully!")
