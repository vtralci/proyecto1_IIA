import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.base
import sklearn.compose
import sklearn.impute
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble
import signal   
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score


# ── Feature groups ────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "approximate_latitude", "approximate_longitude", "size", "floor", "land_size",
    "nb_rooms", "nb_bedrooms", "nb_bathrooms", "nb_parking_places", "nb_boxes", "nb_terraces",
]
BINARY_FEATURES = [
    "has_a_balcony", "has_a_cellar", "has_a_garage",
    "has_air_conditioning", "last_floor", "upper_floors",
    "floor_is_missing", "land_size_is_missing", "nb_bathrooms_is_missing",  # ← added for MissingnessFlagger
]
OHE_FEATURES = ["property_type"]
TARGET_ENC_FEATURES = ["city"]
POSTAL_FEATURE = ["postal_code"]


# ── Custom transformer: postal code → département ─────────────────────────────

class DepartmentExtractor(skl.base.BaseEstimator, skl.base.TransformerMixin):
    
    """
    Extracts the French département code from a postal code.
    '75013' → '75',  '2A004' → '2A' (Corse)
    Falls back to 'unknown' for nulls or malformed values.
    """

    def fit(self, X, y=None):
        #
        return self

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        def extract(code):
            code = str(code).strip()
            if len(code) < 2 or code.lower() == "nan":
                return "unknown"
            # Corse special cases: 2A / 2B
            if code[:2].upper() in ("2A", "2B"):
                return code[:2].upper()
            return code[:2]

        return np.array([extract(c) for c in X[:, 0]]).reshape(-1, 1)
    def get_feature_names_out(self, input_features=None):   
        return np.array(["department"])

PROPERTY_TYPE_MAP = {
    "appartement": "apartment", "duplex": "apartment", "loft": "apartment",
    "chambre": "apartment", "péniche": "apartment",
    "maison": "house", "villa": "house", "ferme": "house",
    "chalet": "house", "gîte": "house", "atelier": "house",
    "château": "prestige", "manoir": "prestige", "propriété": "prestige",
    "hôtel particulier": "prestige", "moulin": "prestige", "hôtel": "prestige",
    "terrain": "land", "terrain à bâtir": "land",
    "divers": "other", "viager": "other", "parking": "other",
}

class PropertyTypeGrouper(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, mapping=PROPERTY_TYPE_MAP):
        self.mapping = mapping

    def fit(self, X, y=None):
        # Store input feature names for get_feature_names_out
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X = X.copy()
        X["property_type"] = X["property_type"].map(self.mapping).fillna("other")
        return X

    def get_feature_names_out(self, input_features=None):
        # Passthrough: same columns in, same columns out (only values change)
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.array([], dtype=object))


class MissingnessFlagger(skl.base.BaseEstimator, skl.base.TransformerMixin):
    FLAG_COLS = ["floor", "land_size", "nb_bathrooms"]

    def fit(self, X, y=None):
        # Store input feature names before we add the flag columns
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.FLAG_COLS:
            X[f"{col}_is_missing"] = X[col].isna().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        # Returns original features PLUS the 3 new flag columns
        base = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else getattr(self, "feature_names_in_", np.array([], dtype=object))
        )
        new_cols = np.array([f"{col}_is_missing" for col in self.FLAG_COLS], dtype=object)
        return np.concatenate([base, new_cols])
# ── Sub-skl.pipeline.Pipelines ─────────────────────────────────────────────────────────────

def _numeric_pipe():
    """Median imputation (robust to outliers in real estate data)."""
    return skl.pipeline.Pipeline([
        ("imputer", skl.impute.SimpleImputer(strategy="median")),
    ])


def _binary_pipe():
    """Fill missing binary flags with 0 (feature absent = no)."""
    return skl.pipeline.Pipeline([
        ("imputer", skl.impute.SimpleImputer(strategy="constant", fill_value=0)),
    ])


def _ohe_pipe():
    """One-hot encoding for low-cardinality categoricals."""
    return skl.pipeline.Pipeline([
        ("imputer", skl.impute.SimpleImputer(strategy="most_frequent")),
        ("ohe", skl.preprocessing.OneHotEncoder(
            handle_unknown="ignore",   # unseen categories → all zeros
            sparse_output=False,
            min_frequency=10,          # rare categories → 'infrequent_sklearn'
        )),
    ])

def _target_enc_pipe():
    """
    Smoothed target encoding for city (8643 unique values).
    Shrinks rare cities toward the global mean to prevent overfitting.
    Only fitted on training data — no leakage.
    """
    return skl.pipeline.Pipeline([
        ("imputer", skl.impute.SimpleImputer(strategy="constant", fill_value="unknown")),
        ("target_enc", skl.preprocessing.TargetEncoder(
            smooth="auto",      # sklearn auto-computes shrinkage per category
            target_type="continuous",
            cv=5,               # internal cross-val to reduce in-fold leakage
        )),
    ])


def _department_pipe():
    """Postal code → département string → one-hot (≤ 96 + unknown)."""
    return skl.pipeline.Pipeline([
        ("extract_dept", DepartmentExtractor()),
        ("ohe", skl.preprocessing.OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])


# ── Main preprocessor ─────────────────────────────────────────────────────────

def build_preprocessor() -> skl.pipeline.Pipeline:
    """
    Now returns a Pipeline (not bare ColumnTransformer) so that
    PropertyTypeGrouper and MissingnessFlagger run first, before
    the ColumnTransformer sees the data.
    """
    column_transformer = skl.compose.ColumnTransformer(
        transformers=[
            ("num",        _numeric_pipe(),    NUMERIC_FEATURES),
            ("bin",        _binary_pipe(),     BINARY_FEATURES),   # includes *_is_missing cols
            ("ohe",        _ohe_pipe(),        OHE_FEATURES),
            ("city",       _target_enc_pipe(), TARGET_ENC_FEATURES),
            ("department", _department_pipe(), POSTAL_FEATURE),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        n_jobs=1,
    )
    return skl.pipeline.Pipeline([
        ("group_property_type", PropertyTypeGrouper()),  # 22 types → 6
        ("flag_missing",        MissingnessFlagger()),   # adds *_is_missing cols
        ("column_transformer",  column_transformer),
    ])
# ── Full model skl.pipeline.Pipeline (preprocessor + RF) ───────────────────────────────────

def build_pipeline() -> skl.pipeline.Pipeline:
    """
    Returns the full skl.pipeline.Pipeline: preprocessing → RandomForestRegressor.
    Tune hyperparameters via GridSearchCV or RandomizedSearchCV.
    """
    
    return skl.pipeline.Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", sklearn.ensemble.RandomForestRegressor(
            n_estimators=300,
            max_features=0.3,       # ~30 % of features per split (good default for RF)
            min_samples_leaf=3,     # prevents overfitting on rare cities
            n_jobs=1,
            random_state=42,
        )),
    ])



if __name__ == "__main__":
 
    # ── Load and join on shared key ───────────────────────────────────────────
    KEY = "id_annonce"
    TARGET = "price"
 
    X_raw = pd.read_csv("Data/X_train.csv")
    y_raw = pd.read_csv("Data/Y_train.csv")        # must contain KEY + TARGET columns
 
    # Merge on id_annonce — safe against different orderings in the two files
    df = X_raw.merge(y_raw[[KEY, TARGET]], on=KEY, how="inner", validate="1:1")
 
    print(f"Loaded {len(df)} rows after joining on '{KEY}' "
          f"(X: {len(X_raw)}, y: {len(y_raw)})")
 
    # AFTER — exclude the 3 flag cols (MissingnessFlagger creates them at runtime)
    MISSING_FLAG_COLS = ["floor_is_missing", "land_size_is_missing", "nb_bathrooms_is_missing"]

    FEATURES = (
        NUMERIC_FEATURES + 
        [f for f in BINARY_FEATURES if f not in MISSING_FLAG_COLS] +  # ← don't pre-select flags
        OHE_FEATURES + TARGET_ENC_FEATURES + POSTAL_FEATURE
    )   
    X = df[FEATURES]
    y = df[TARGET]
    
    print(f"Final feature set: {len(X.columns)} columns")
 
    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    
    # ── Fit & evaluate ────────────────────────────────────────────────────────
    pipeline = build_pipeline()
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)

    print("Shape:", X_train_transformed.shape)
    print("dtype:", X_train_transformed.dtype)
    print("NaNs:", np.isnan(X_train_transformed).sum())
    print("Infs:", np.isinf(X_train_transformed).sum())
    print("Sample:\n", X_train_transformed[:3])
    print("y_train dtype:", y_train.dtype)
    print("y_train NaNs:", y_train.isna().sum())
    print("y_train Infs:", np.isinf(y_train).sum())
    print("y_train sample:", y_train[:5].values)
    
    rf_pipeline = build_pipeline()
    
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # ignore interrupts
        print("Fitting pipeline... (press Ctrl+C to skip fitting and evaluate with unfitted model)")
        rf_pipeline.fit(X_train, y_train)
        print("Pipeline fitted successfully.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)   # restore
    except Exception as e:
        print("Error during fitting:", e)
        print(type(e).__name__, ":", e)
   
    print("Evaluating with cross-validation...")
 
    scores = cross_val_score(
        rf_pipeline, X_train, y_train,
        scoring="r2", cv=10, n_jobs=1
    )
    
    print(f"Cross-val R² scores: {scores}")
    print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")
    y_pred = rf_pipeline.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"R²   : {r2:.3f}")       # % variance explained. 1.0 = perfect
    print(f"MAE  : {mae:,.0f} €")   # average absolute error
    print(f"RMSE : {rmse:,.0f} €")  # penalises large errors more than MAE
    print(f"MAPE : {mape:.1f} %")   # error as % of actual price
    
    y_pred = rf_pipeline.predict(X_test)
    residuals = y_test.values - y_pred

    # Predicted vs Actual
    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual (€)", "y": "Predicted (€)"},
                    title="Predicted vs Actual", opacity=0.3)
    fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(),
                y0=y_test.min(), y1=y_test.max(), line=dict(color="red", dash="dash"))
    fig.write_html("predicted_vs_actual.html")

    # Residuals
    fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted (€)", "y": "Residual (€)"},
                    title="Residuals vs Predicted", opacity=0.3)
    fig.add_hline(y=0, line_color="red", line_dash="dash")
    fig.write_html("residuals.html")

    # Feature importances
    feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances   = rf_pipeline.named_steps["model"].feature_importances_
    top20_idx     = np.argsort(importances)[-20:]

    fig = px.bar(x=importances[top20_idx],
                y=[feature_names[i] for i in top20_idx],
                orientation="h", title="Top 20 Feature Importances",
                labels={"x": "Importance", "y": "Feature"})
    fig.write_html("feature_importances.html")

    # Residuals distribution
    fig = px.histogram(x=residuals, nbins=50, title="Residuals Distribution",
                    labels={"x": "Residual (€)"})
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.write_html("residuals_distribution.html")

    print("All charts saved as .html files — open them in your browser.")
    
# Example prediction on a new property
    new_property = pd.DataFrame([{
    "approximate_latitude":  48.8396,
    "approximate_longitude": 2.2961,
    "size":                  8322,
    "floor":                 6,
    "land_size":             0,
    "energy_performance_value": 224,
    "ghg_value":             52,
    "nb_rooms":              5,
    "nb_bedrooms":           3,
    "nb_bathrooms":          1,
    "nb_parking_places":     0,
    "nb_boxes":              0,
    "nb_photos":             9,
    "nb_terraces":           0,
    "has_a_balcony":         1,
    "has_a_cellar":          0,
    "has_a_garage":          0,
    "has_air_conditioning":  0,
    "last_floor":            0,
    "upper_floors":          0,
    "property_type":         "apartment",
    "exposition":            None,
    "energy_performance_category": "D",
    "ghg_category":          "E",
    "city":                  "paris-15eme",
    "postal_code":           "75015",
}])

MISSING_FLAG_COLS = ["floor_is_missing", "land_size_is_missing", "nb_bathrooms_is_missing"]

float_cols = NUMERIC_FEATURES + [f for f in BINARY_FEATURES if f not in MISSING_FLAG_COLS]
new_property[float_cols] = new_property[float_cols].astype(float)

predicted_price = rf_pipeline.predict(new_property)

predicted_price = rf_pipeline.predict(new_property)
print(f"Predicted price: {predicted_price[0]:,.0f} €")