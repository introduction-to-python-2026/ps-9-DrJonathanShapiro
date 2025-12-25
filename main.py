import pandas as pd
import joblib


def read_config_yaml(path="config.yaml"):
    """
    Minimal YAML reader for exactly this format:

    selected_features: ["PPE", "spread1"]
    path: "parkinsons_model.joblib"
    """
    selected_features = None
    model_path = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("selected_features:"):
                # everything after the colon is the list
                rhs = line.split(":", 1)[1].strip()
                # rhs looks like ["PPE", "spread1"]
                rhs = rhs.strip().lstrip("[").rstrip("]")
                parts = [p.strip().strip('"').strip("'") for p in rhs.split(",") if p.strip()]
                selected_features = parts

            if line.startswith("path:"):
                rhs = line.split(":", 1)[1].strip()
                model_path = rhs.strip('"').strip("'")

    if selected_features is None or model_path is None:
        raise ValueError("config.yaml must contain selected_features and path")

    return selected_features, model_path


def main():
    # Read config.yaml
    selected_features, model_path = read_config_yaml("config.yaml")

    # Load dataset
    df = pd.read_csv("parkinsons.csv")

    # Load trained model bundle
    bundle = joblib.load(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]

    # (target_col may or may not exist in bundle; not needed for predicting)
    X = df[selected_features]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    # Keep output minimal
    print(len(preds))


if __name__ == "__main__":
    main()
