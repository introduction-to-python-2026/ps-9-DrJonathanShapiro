import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def main():
    df = pd.read_csv("parkinsons.csv")

    # features must match config.yaml
    X = df[["PPE", "spread1"]]
    y = df["status"].astype(int)

    # one object that scales + predicts
    model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    model.fit(X, y)

    # IMPORTANT: save ONLY the model (not a dict)
    joblib.dump(model, "parkinsons_model.joblib")
    print("Saved parkinsons_model.joblib")

if __name__ == "__main__":
    main()
