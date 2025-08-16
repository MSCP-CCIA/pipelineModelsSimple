"""
Iris Classification Pipeline with Hyperparameter Optimization
-------------------------------------------------------------

This script builds a professional machine learning pipeline for the Iris dataset.
The pipeline includes preprocessing, feature selection, dimensionality reduction,
and model training with hyperparameter tuning using GridSearchCV.

It saves both:
1. The optimized pipeline as a `.joblib` file.
2. A JSON file with metadata about the training (parameters, scores, environment).

Author: Manuel Castro
Date: [2025-08-16]
"""

# ========== Imports ==========
import logging
import joblib
import json
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn
import numpy


# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,  # INFO for production, DEBUG for dev
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


# ========== Step 1: Load Dataset ==========
iris = load_iris()
X, y = iris.data, iris.target
logger.info("Loaded Iris dataset with %d samples and %d features.", X.shape[0], X.shape[1])


# ========== Step 2: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info("Split dataset into train (%d) and test (%d).", X_train.shape[0], X_test.shape[0])


# ========== Step 3: Define Pipeline ==========
base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", HistGradientBoostingClassifier(random_state=42))
])


# ========== Step 4: Define Hyperparameter Grid ==========
param_grid = {
    "pca__n_components": [None, 2, 3, 4],
    "select__k": [2, 3, 4],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__max_depth": [None, 3, 5],
    "clf__l2_regularization": [0.0, 1.0]
}
logger.info("Hyperparameter grid defined with %d combinations.",
            np.prod([len(v) for v in param_grid.values()]))


# ========== Step 5: Cross-Validation Strategy ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ========== Step 6: Grid Search Optimization ==========
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


# ========== Step 7: Report Best Results ==========
logger.info("Best Parameters: %s", grid_search.best_params_)
logger.info("Best Cross-Validation Accuracy: %.4f", grid_search.best_score_)


# ========== Step 8: Retrieve Best Pipeline ==========
best_pipeline = grid_search.best_estimator_
logger.info("Best pipeline retrieved.")


# ========== Step 9: Evaluate on Test Set ==========
y_pred = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
logger.info("Test Set Accuracy: %.4f", test_accuracy)
logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
logger.info("Classification Report:\n%s",
            classification_report(y_test, y_pred, target_names=iris.target_names))


# ========== Step 10: Save Optimized Pipeline ==========
model_path = "iris_pipeline.joblib"
joblib.dump(best_pipeline, model_path)
logger.info("Optimized pipeline saved as '%s'", model_path)


# ========== Step 11: Save Metadata as JSON ==========
metadata = {
    "model_file": model_path,
    "trained_on": datetime.now().isoformat(),
    "dataset": "Iris",
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

with open("iris_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logger.info("Metadata saved to 'iris_metadata.json'")
