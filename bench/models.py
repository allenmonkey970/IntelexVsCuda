"""Model registry: one hyperparameter set per model, shared by every dataset and implementation.

Estimators are imported inside build() on purpose. patch_sklearn() only replaces the
classes in the sklearn modules, so a class imported before patching stays unpatched.
Importing here, after benchmark.py has applied the patch, guarantees Intelex runs
really use the sklearnex implementations.
"""
RANDOM_STATE = 42

# Only models that scikit-learn-intelex actually patches, so every Intelex row reflects a
# real oneDAL acceleration (verified against sklearnex.get_patch_map()). LightGBM/XGBoost
# (not sklearn) and AdaBoost/MLP (sklearn but unpatched) were removed for that reason.
MODELS = [
    "Random Forest Classifier",
    "Logistic Regression",
    "Extra Trees Classifier",
]

# Models whose inputs are standardised (fitted on the training split) before fitting.
NEEDS_SCALING = {"Logistic Regression"}

# Models that can consume CuPy arrays directly and train on the GPU. None of the patched
# sklearn estimators have a GPU path, so under CuPY they run on the host after a round-trip.
GPU_NATIVE = set()


def build(name, implementation):
    if name == "Random Forest Classifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if name == "Extra Trees Classifier":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")
