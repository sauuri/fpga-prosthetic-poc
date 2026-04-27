import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle


def train_classifier(X, y, model_type='svm', tune=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == 'svm':
        if tune:
            print("  GridSearch 튜닝 중 (시간 소요)...")
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf')),
            ])
            param_grid = {'svm__C': [1, 10, 100], 'svm__gamma': ['scale', 0.01, 0.001]}
            model = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
        else:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=10)),
            ])
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    if tune and model_type == 'svm':
        print(f"  최적 파라미터: {model.best_params_}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{model_type}] Test accuracy: {acc:.4f}")
    return model, acc


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
