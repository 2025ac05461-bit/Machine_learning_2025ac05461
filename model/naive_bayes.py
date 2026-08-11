import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()
X, y = data.data, data.target

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GaussianNB())
])
model.fit(X_train, y_train)
joblib.dump(model, "naive_bayes.pkl")
print("Saved naive_bayes.pkl")
