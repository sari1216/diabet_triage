import joblib
import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from fastapi import FastAPI


Xy = load_diabetes(as_frame=True)

X = Xy.frame.drop(columns=["target"])

y = Xy.frame["target"] # acts as a "progression index" (higher = worse)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Skapa pipelinen med de två stegen
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Träna modellen (skalar X_train och tränar sedan på det)
pipeline.fit(X_train, y_train)
# 1. Gör förutsägelser på testdatan
y_pred = pipeline.predict(X_test)

# 2. Beräkna Root Mean Squared Error (RMSE)
rmse = root_mean_squared_error(y_test, y_pred)

# 3. Skriv ut resultatet
print("RMSE:", rmse)
joblib.dump(pipeline, "pipeline.joblib")
metrics = {"model_version": "v0.1", "rmse": rmse}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


