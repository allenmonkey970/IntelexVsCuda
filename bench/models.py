"""Model registry: one hyperparameter set per model, shared by every dataset and implementation.

Estimators are imported inside build() on purpose. patch_sklearn() only replaces the
classes in the sklearn modules, so a class imported before patching stays unpatched.
Importing here, after benchmark.py has applied the patch, guarantees Intelex runs
really use the sklearnex implementations.
"""
RANDOM_STATE = 42

MODELS = [
    "LightGBM classifier",
    "XGBoost",
    "Logistic Regression",
    "Random Forest Classifier",
    "AdaBoost",
    "Multi-layer Perceptron classifier",
]

# Models whose inputs are standardised (fitted on the training split) before fitting.
NEEDS_SCALING = {"Logistic Regression", "Multi-layer Perceptron classifier"}

# Models that can consume CuPy arrays directly and train on the GPU.
# Everything else must copy CuPy data back to the host before fitting.
GPU_NATIVE = {"XGBoost"}


def build(name, implementation):
    if name == "LightGBM classifier":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        # CPU for BareBones/Intelex so the comparison is CPU vs GPU, not GPU vs GPU.
        device = "cuda" if implementation == "CuPY" else "cpu"
        return XGBClassifier(n_estimators=100, tree_method="hist", device=device,
                             random_state=RANDOM_STATE)
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if name == "Random Forest Classifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    if name == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)
    if name == "Multi-layer Perceptron classifier":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1200, learning_rate="adaptive",
                             early_stopping=True, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")
