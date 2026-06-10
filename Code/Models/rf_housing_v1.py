# =============================================================
#  PREDICCIÓN DE PRECIOS DE VIVIENDAS — Random Forest
#  VERSION 1
#
#  Pipeline idéntico al XGBoost v1. Diferencias en el modelo:
#    1. Algoritmo: RandomForestRegressor (sklearn)
#    2. Sin early_stopping — n_estimators fijo en 500
#    3. oob_score=True — validación interna gratuita vía bootstrap
#    4. max_features="sqrt" — equivalente RF de colsample_bytree
#    5. Importancia: Mean Decrease Impurity (misma API .feature_importances_)
#    6. Sin StandardScaler ni eval_set (no aplican a RF)
# =============================================================

import os
import joblib
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from diagnostics import mae_por_decil, plots_trees

ruta = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------
# 1. CARGAR LOS DATOS
# ------------------------------------------------------------------
df = pd.read_csv(os.path.join(ruta, "Data/trainData.csv"))

print("=== DATOS CARGADOS ===")
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(f"Precio medio inicial: {df['price'].mean():,.0f} €\n")

def log_filtro(df_antes, df_despues, descripcion):
    eliminadas = len(df_antes) - len(df_despues)
    pct = eliminadas / len(df_antes) * 100
    print(f"[FILTRO] {descripcion}: -{eliminadas} filas ({pct:.1f}%)")
    return df_despues

# ------------------------------------------------------------------
# 2. AGRUPAR TIPOS DE VIVIENDA EN 3 CATEGORÍAS Y FILTRAR
# ------------------------------------------------------------------
PROPERTY_TYPE_MAP = {
    "appartement": ["appartement", "duplex", "loft"],
    "maison":      ["maison", "ferme", "chalet"],
    "lujo":        ["villa", "propriété"],
}
tipo_a_grupo = {t: g for g, ts in PROPERTY_TYPE_MAP.items() for t in ts}
df["property_group"] = df["property_type"].map(tipo_a_grupo)

n_antes = len(df)
df = df[df["property_group"].notna()].copy()
print(f"Filas eliminadas por tipo no mapeado: {n_antes - len(df)}")
print(df["property_group"].value_counts(), "\n")

df = log_filtro(df, df[df["property_group"].notna()].copy(), "property_group no mapeado")

# ------------------------------------------------------------------
# 3. ELIMINAR OUTLIERS DE PRECIO (percentil 99)
# ------------------------------------------------------------------
precio_p99 = df["price"].quantile(0.99)
df = df[df["price"] <= precio_p99].copy()
print(f"Precio máximo tras cap p99: {precio_p99:,.0f} €\n")
df = log_filtro(df, df[df["price"] <= precio_p99].copy(), "outliers precio p99")

# ------------------------------------------------------------------
# 4. FILTRAR TAMAÑOS IMPOSIBLES
# ------------------------------------------------------------------
df = df[(df["size"] >= 10) & (df["size"] <= 5000)].copy()
print(f"Filas tras filtro de size [10, 5000 m²]: {len(df)}\n")
df = log_filtro(df, df[(df["size"] >= 10) & (df["size"] <= 5000)].copy(), "size fuera de [10, 5000]")
df = df[~(df["nb_bedrooms"] > df["nb_rooms"])].copy() 

# ------------------------------------------------------------------
# 5. FILTRAR ESCENARIOS IMPOSIBLES
# ------------------------------------------------------------------
df["has_energy_cert"] = df["energy_performance_category"].notna().astype(int)
df["has_ghg_value"] = df["ghg_value"].notna().astype(int)
df = df.drop(columns=["energy_performance_category", "energy_performance_value", "ghg_value", "ghg_category"])

# ------------------------------------------------------------------
# 6. VARIABLES ELIMINADAS POR EXCESO DE NULOS (>50%)
# ------------------------------------------------------------------
COLS_ALTO_NULO = [
    "exposition", "floor", "land_size", "nb_bathrooms"
]
df = df.drop(columns=COLS_ALTO_NULO)

# ------------------------------------------------------------------
# 7. DEPARTAMENTO DESDE CÓDIGO POSTAL
# ------------------------------------------------------------------
df["dept"] = df["postal_code"].astype(str).str.zfill(5).str[:2]
df["dept"] = df["dept"].replace({"2A": "20", "2B": "20"})

dept_counts = df["dept"].value_counts()
depts_validos = dept_counts[dept_counts >= 30].index
n_antes = len(df)
df = df[df["dept"].isin(depts_validos)].copy()
print(f"Departamentos conservados: {df['dept'].nunique()} "
      f"(eliminadas {n_antes - len(df)} filas de depts con <30 anuncios)\n")

# Label-encode dept como entero (Random Forest no necesita OHE)
dept_categorias = sorted(df["dept"].unique())
dept_to_int = {d: i for i, d in enumerate(dept_categorias)}
df["dept_enc"] = df["dept"].map(dept_to_int)
df = log_filtro(df, df[df["dept"].isin(depts_validos)].copy(), "depts con <100 anuncios")


# ------------------------------------------------------------------
# 8. ELIMINAR COLUMNAS NO NECESARIAS
# ------------------------------------------------------------------
df = df.drop(columns=[
    "id_annonce", "city", "postal_code",
    "property_type", "dept"
])

# ------------------------------------------------------------------
# 9. ELIMINAR FILAS CON CUALQUIER NULO RESTANTE (sin imputación)
# ------------------------------------------------------------------
n_antes = len(df)
nulos_por_col = df.isnull().sum()
nulos_por_col = nulos_por_col[nulos_por_col > 0].sort_values(ascending=False)
print("Columnas con nulos antes del dropna:")
print(nulos_por_col)

# Filas que se eliminarán (tienen al menos 1 nulo)
mask_nulos = df.isnull().any(axis=1)
print(f"\nFilas afectadas: {mask_nulos.sum()}")
print("Distribución de cuántos nulos tiene cada fila afectada:")
print(df[mask_nulos].isnull().sum(axis=1).value_counts().sort_index())

df = df.dropna()
print(f"Filas eliminadas por nulos restantes: {n_antes - len(df)}")
df = log_filtro(df, df.dropna(), "filas con nulos restantes")

print(f"Filas finales limpias: {len(df)}")
print(f"Nulos totales: {df.isnull().sum().sum()}\n")

# ------------------------------------------------------------------
# 10. DISTANCIA A CIUDADES PRINCIPALES (km, haversine) — VECTORIZADO
# ------------------------------------------------------------------
CIUDADES = {
    "dist_paris":     (48.8566,  2.3522),
    "dist_lyon":      (45.7640,  4.8357),
    "dist_marseille": (43.2965,  5.3698),
    "dist_toulouse":  (43.6047,  1.4442),
    "dist_bordeaux":  (44.8378, -0.5792),
    "dist_nice":      (43.7102,  7.2620),
    "dist_nantes":    (47.2184, -1.5536),
    "dist_lille":     (50.6292,  3.0573),
}

def haversine_vec(lat1, lon1, lat2, lon2):
    """Haversine vectorizado — lat1/lon1 son arrays NumPy."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

lat = df["approximate_latitude"].values
lon = df["approximate_longitude"].values

for nombre, (clat, clon) in CIUDADES.items():
    df[nombre] = haversine_vec(lat, lon, clat, clon)

df["dist_min_ciudad"] = df[[*CIUDADES]].min(axis=1)
# ------------------------------------------------------------------
# 11. LOG-TRANSFORM EN SIZE
# ------------------------------------------------------------------
df["log_size"] = np.log1p(df["size"])
df = df.drop(columns=["size"])

# ------------------------------------------------------------------
# 12. LABEL ENCODING DE property_group
#     Random Forest no necesita OHE — un entero por categoría es suficiente
# ------------------------------------------------------------------
grupo_to_int = {"appartement": 0, "maison": 1, "lujo": 2}
df["property_group_enc"] = df["property_group"].map(grupo_to_int)
df = df.drop(columns=["property_group"])

print("=== VARIABLES USADAS EN EL MODELO ===")
feature_cols = [c for c in df.columns if c != "price"]
print(feature_cols)
print(f"\nTotal variables: {len(feature_cols)}\n")

# ------------------------------------------------------------------
# 13. EXPORTAR DATOS PROCESADOS
# ------------------------------------------------------------------
print(f"Dataset final: {len(df)} filas, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ------------------------------------------------------------------
# 14. SEPARAR FEATURES Y OBJETIVO
#     Target: log1p(price) — Random Forest converge mejor en espacio log
#     No se necesita StandardScaler adicional
# ------------------------------------------------------------------
X = df.drop(columns=["price"])
y_log = np.log1p(df["price"].values)

X_train, X_test, y_train, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test = np.expm1(y_test_log)   # precio real para métricas finales

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ------------------------------------------------------------------
# 15. ENTRENAR RANDOM FOREST
#     Hiperparámetros razonados:
#       n_estimators=500     — suficiente para estabilizar el ensemble;
#                              RF converge (↓ error) con más árboles pero
#                              el retorno marginal es mínimo tras ~300
#       max_depth=None       — árboles completos; la regularización viene
#                              del bagging, no de la profundidad
#       max_features="sqrt"  — equivalente RF de colsample_bytree:
#                              en cada split se evalúan √p features
#                              (estándar para regresión, reduce correlación
#                              entre árboles)
#       min_samples_leaf=4   — un nodo hoja necesita ≥4 muestras;
#                              evita memorizar outliers individuales
#       bootstrap=True       — muestras con reemplazo (bagging estándar)
#       oob_score=True       — estimación gratuita del error de generalización
#                              usando las muestras out-of-bag de cada árbol
#       n_jobs=-1            — paralelización completa
# ------------------------------------------------------------------
modelo = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=4,
    bootstrap=True,
    oob_score=False,   # desactivado: causa deadlock en Python 3.13
    random_state=42,
    n_jobs=-1,          # single-thread: evita bug joblib/_pre_dispatch_amount en Python 3.13
    verbose=0,         # verbose>0 activa print_progress que falla con este joblib
)

print("=== ENTRENANDO RANDOM FOREST ===")
print(f"Entrenando {modelo.n_estimators} árboles en single-thread, puede tardar 3-5 min...")
modelo.fit(X_train, y_train)
print("Entrenamiento completado.")

# ------------------------------------------------------------------
# 16. EVALUAR EL MODELO
# ------------------------------------------------------------------
y_pred_log = modelo.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_pred     = np.clip(y_pred, df["price"].min(), df["price"].max())

mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("=== RESULTADOS ===")
print(f"MAE   (Error Absoluto Medio)       : {mae:,.0f} €")
print(f"  → En promedio, nos equivocamos en {mae:,.0f} € por vivienda")
print(f"MAPE  (Error Porcentual Absoluto)   : {mape:.1f}%")
print(f"R²    (Coeficiente determinación)   : {r2:.4f}")
print(f"  → El modelo explica el {r2*100:.1f}% de la variación en precios")
# OOB score desactivado (ver comentario en bloque de entrenamiento)

# ------------------------------------------------------------------
# 17. DIAGNÓSTICO COMPLETO
# ------------------------------------------------------------------
GROUP_MAP = {0: "appartement", 1: "maison", 2: "lujo"}

df_decil = plots_trees( 
    y_true        = y_test,
    y_pred        = y_pred,
    nombre        = "Random Forest",
    out_dir       = ruta,
    modelo        = modelo,
    feature_names = list(X_train.columns),
    group_labels  = X_test["property_group_enc"].values,
    group_map     = GROUP_MAP,
)

joblib.dump(modelo, 'models_pkl/random_forest.pkl')
joblib.dump(list(X_train.columns), 'models_pkl/random_forest_feature_cols.pkl')

os.makedirs('outputs', exist_ok=True)
pd.DataFrame([{"RMSE": np.sqrt(np.mean((y_test - y_pred)**2)), "MAE": mae, "R2": r2}]).to_csv('outputs/metrics_rf.csv', index=False)
pd.DataFrame({"y_real": y_test, "y_pred": y_pred}).to_csv('outputs/predictions_rf.csv', index=False)