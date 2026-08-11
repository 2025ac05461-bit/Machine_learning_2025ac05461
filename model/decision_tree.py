import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
data = load_breast_cancer()
X, y = data.data, data.target

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

model = DecisionTreeClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)
joblib.dump(model, "decision_tree.pkl")
print("Saved decision_tree.pkl")
