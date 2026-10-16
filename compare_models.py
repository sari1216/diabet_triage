from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression (v0.1)": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "LassoCV": LassoCV(cv=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
}

for name, model in models.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", model)])
    pipeline.fit(X_train, y_train)
    rmse = root_mean_squared_error(y_test, pipeline.predict(X_test))
    print(f"{name:25s} RMSE: {rmse:.2f}")