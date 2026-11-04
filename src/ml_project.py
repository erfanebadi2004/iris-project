import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
feature_name = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

df_iris = pd.DataFrame(data, columns=feature_name)

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(df_iris, y, test_size=0.2, random_state=42)

X_test.to_csv("model_test.csv", index=False)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

print("Training model...")
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model.joblib")
print("Model saved successfully as 'model.joblib'.")