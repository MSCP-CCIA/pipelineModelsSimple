"""
Titanic Classification Pipeline with Hyperparameter Optimization
---------------------------------------------------------------

This script builds a professional machine learning pipeline for the Titanic dataset
with selected columns: ['survived', 'pclass', 'who', 'age'].
The pipeline includes preprocessing, model training, and hyperparameter tuning
using GridSearchCV.

It saves both:
1. The optimized pipeline as a `.joblib` file.
2. A JSON file with metadata about the training (parameters, scores, environment).

Author: Andres Hurtado
Date: [2025-08-18]
"""

# ========== Imports ==========
import logging
import joblib
import json
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn
import numpy


# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


# ========== Step 1: Load Dataset ==========
data = sns.load_dataset("titanic")
data = data[['survived', 'pclass', 'who', 'age']]

X = data.drop(columns=['survived'])
y = data['survived']
logger.info("Loaded Titanic dataset with %d samples and %d features.", X.shape[0], X.shape[1])


# ========== Step 2: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info("Split dataset into train (%d) and test (%d).", X_train.shape[0], X_test.shape[0])


# ========== Step 3: Define Preprocessing ==========
numeric_features = ['age']
categorical_features = ['pclass', 'who']

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ========== Step 4: Define Pipeline ==========
base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])


# ========== Step 5: Define Hyperparameter Grid ==========
param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l2", "l1"],
    "clf__solver": ["saga","liblinear"]
}
logger.info("Hyperparameter grid defined with %d combinations.",
            np.prod([len(v) for v in param_grid.values()]))


# ========== Step 6: Cross-Validation Strategy ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ========== Step 7: Grid Search Optimization ==========
grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
logger.info("Starting grid search...")
grid_search.fit(X_train, y_train)
logger.info("Grid search completed.")


# ========== Step 8: Report Best Results ==========
logger.info("Best Parameters: %s", grid_search.best_params_)
logger.info("Best Cross-Validation Accuracy: %.4f", grid_search.best_score_)


# ========== Step 9: Retrieve Best Pipeline ==========
best_pipeline = grid_search.best_estimator_
logger.info("Best pipeline retrieved.")


# ========== Step 10: Evaluate on Test Set ==========
y_pred = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
logger.info("Test Set Accuracy: %.4f", test_accuracy)
logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))


# ========== Step 11: Save Optimized Pipeline ==========
model_path = "titanic_pipeline.joblib"
joblib.dump(best_pipeline, model_path)
logger.info("Optimized pipeline saved as '%s'", model_path)


# ========== Step 12: Save Metadata as JSON ==========
metadata = {
    "model_file": model_path,
    "trained_on": datetime.now().isoformat(),
    "dataset": "Titanic (subset: ['pclass','who','age'])",
    "n_samples_train": X_train.shape[0],
    "n_samples_test": X_test.shape[0],
    "best_params": grid_search.best_params_,
    "cv_accuracy": grid_search.best_score_,
    "test_accuracy": test_accuracy,
    "library_versions": {
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__
    }
}

with open("titanic_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logger.info("Metadata saved to 'titanic_metadata.json'")
