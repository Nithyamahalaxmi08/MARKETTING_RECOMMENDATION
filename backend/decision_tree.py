import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_data(df):
    y = df['primary_platform']

    X = df.drop(columns=['product_name', 'primary_platform', 'secondary_platform'])

    cat_cols = ['category', 'brand']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[cat_cols])

    X_numeric = X.drop(columns=cat_cols)
    X_final = np.hstack([X_numeric.values, X_encoded])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X_final, y_encoded, encoder, le


# -------------------------------
# Top-2 Accuracy
# -------------------------------
def top_2_accuracy(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    correct = 0

    for i in range(len(y_test)):
        top2 = np.argsort(probs[i])[-2:]
        if y_test[i] in top2:
            correct += 1

    return correct / len(y_test)


# -------------------------------
# Evaluation Function (UPDATED)
# -------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, le):

    print("\n📊 MODEL EVALUATION")

    # Predictions
    y_pred = model.predict(X_test)

    # Convert to class names
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Overfitting Check
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print("\n🔍 Overfitting Check:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

    # Top-2 Accuracy
    top2_acc = top_2_accuracy(model, X_test, y_test)
    print(f"\n🔥 Top-2 Accuracy: {top2_acc:.4f}")

    # Classification Report (with class names)
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    # Confusion Matrix (with labels)
    print("\n📊 Confusion Matrix:\n")
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    # Cross Validation
    print("\n🔁 Cross Validation (5-Fold):")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Scores:", cv_scores)
    print("Average CV Accuracy:", round(cv_scores.mean(), 4))


# -------------------------------
# Train Decision Tree
# -------------------------------
def train_decision_tree(X, y, le):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🌳 Training Decision Tree...")

    start_time = time.time()

    model = DecisionTreeClassifier(
        criterion='gini',   # or 'entropy'
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"⏱️ Training Time: {training_time:.4f} seconds")

    # Evaluate
    evaluate_model(model, X_train, X_test, y_train, y_test, le)

    return model


# -------------------------------
# Main Function
# -------------------------------
def main():
    print("📂 Loading dataset...")
    df = pd.read_csv("marketing_dataset.csv")
    print("✅ Dataset loaded:", df.shape)

    X, y, encoder, le = preprocess_data(df)

    # Print class mapping
    print("\n📌 Class Mapping:")
    for i, label in enumerate(le.classes_):
        print(f"{i} → {label}")

    model = train_decision_tree(X, y, le)

    # Save model and encoders
    joblib.dump(model, "decision_tree_model.pkl")
    joblib.dump(encoder, "onehot_encoder_dt.pkl")
    joblib.dump(le, "label_encoder_dt.pkl")

    print("\n💾 Decision Tree model saved successfully!")


if __name__ == "__main__":
    main()