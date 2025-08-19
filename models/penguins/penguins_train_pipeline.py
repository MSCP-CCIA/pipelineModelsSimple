"""
Penguins Sex Classification Pipeline
------------------------------------

Pipeline de clasificación para predecir "sex" en el dataset Penguins.
Incluye preprocesamiento de categóricas y numéricas, selección de características,
reducción de dimensionalidad y optimización de hiperparámetros.

Autor: Mateo Fonseca
Fecha: [2025-08-18]
"""

# ========== Imports ==========
import logging
import joblib
import json
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn
import numpy

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_penguins.log")
    ]
)
logger = logging.getLogger(__name__)

# ========== Step 1: Load Dataset ==========
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()  # eliminar filas con valores faltantes

X = penguins.drop("sex", axis=1)
y = penguins["sex"]

# Identificar columnas categóricas y numéricas
categorical_features = ["species", "island"]
numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

logger.info("Dataset loaded with %d samples, %d categorical and %d numerical features.",
            X.shape[0], len(categorical_features), len(numeric_features))

# ========== Step 2: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info("Split dataset into train (%d) and test (%d).", X_train.shape[0], X_test.shape[0])

# ========== Step 3: Define Preprocessor ==========
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ========== Step 4: Define Pipeline ==========
base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", HistGradientBoostingClassifier(random_state=42))
])

# ========== Step 5: Define Hyperparameter Grid ==========
param_grid = {
    "select__k": [2, 3, 4, 5, 6],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__max_depth": [None, 3, 5],
    "clf__l2_regularization": [0.0, 1.0]
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
logger.info("Classification Report:\n%s",
            classification_report(y_test, y_pred, target_names=y.unique()))

# ========== Step 11: Save Optimized Pipeline ==========
model_path = "penguins_pipeline.joblib"
joblib.dump(best_pipeline, model_path)
logger.info("Optimized pipeline saved as '%s'", model_path)

# ========== Step 12: Save Metadata as JSON ==========
metadata = {
    "model_file": model_path,
    "trained_on": datetime.now().isoformat(),
    "dataset": "Penguins (Palmer Archipelago)",
    "target": "sex",
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

with open("penguins_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logger.info("Metadata saved to 'penguins_metadata.json'")
