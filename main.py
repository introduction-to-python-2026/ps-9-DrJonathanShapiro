import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    # Load dataset
    df = pd.read_csv("parkinsons.csv")

    # Target column
    if "status" in df.columns:
        target_col = "status"
    else:
        target_col = [c for c in df.columns if df[c].nunique() == 2][0]

    y = df[target_col].astype(int)

    # Select features (must match config.yaml)
    selected_features = ["PPE", "spread1"]
    X = df[selected_features]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print("Accuracy:", acc)

    # Save model
    bundle = {
        "model": model,
        "scaler": scaler,
        "selected_features": selected_features,
        "target_col": target_col
    }

    joblib.dump(bundle, "parkinsons_model.joblib")
    print("Saved parkinsons_model.joblib")


if __name__ == "__main__":
    main()

