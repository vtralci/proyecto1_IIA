# =============================================================
#  PREDICCIÓN DE PRECIOS DE VIVIENDAS — XGBoost 
# =============================================================

import os
import io
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from diagnostics import plots_trees

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. CARGAR DATOS
# ------------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "Data/trainData.csv"))
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(f"Precio medio inicial: {df['price'].mean():,.0f} €\n")


# ------------------------------------------------------------------
# 2. AGRUPAR TIPOS DE VIVIENDA
# Mapea property_type a tres grupos: appartement, maison, lujo.
# Filas con tipos no reconocidos se descartan.
# ------------------------------------------------------------------
PROPERTY_TYPE_MAP = {
    "appartement": ["appartement", "duplex", "loft"],
    "maison":      ["maison", "ferme", "chalet"],
    "lujo":        ["villa", "propriété"],
}
tipo_a_grupo = {t: g for g, ts in PROPERTY_TYPE_MAP.items() for t in ts}
df["property_group"] = df["property_type"].map(tipo_a_grupo)
df = df[df["property_group"].notna()].copy()


# ------------------------------------------------------------------
# 3. FILTROS DE CALIDAD
# Elimina precios en el percentil 99, tamaños fuera de rango,
# anuncios con más habitaciones que cuartos, y precios/m² imposibles.
# ------------------------------------------------------------------
precio_p99 = df["price"].quantile(0.99)
df = df[df["price"] <= precio_p99].copy()
df = df[(df["size"] >= 10) & (df["size"] <= 5000)].copy()
df = df[~(df["nb_bedrooms"] > df["nb_rooms"])].copy()

df["price_m2_temp"] = df["price"] / df["size"]
df = df[(df["price_m2_temp"] >= 300) & (df["price_m2_temp"] <= 20000)].copy()
df = df.drop(columns=["price_m2_temp"])

print(f"Filas tras filtros de calidad: {len(df)}\n")


# ------------------------------------------------------------------
# 4. ELIMINAR COLUMNAS CON ALTO % DE NULOS
# ------------------------------------------------------------------
df = df.drop(columns=["exposition", "floor", "land_size", "nb_bathrooms"])


# ------------------------------------------------------------------
# 5. CERTIFICADO ENERGÉTICO Y GHG
# Se crean flags binarios de disponibilidad y se eliminan los valores raw.
# ------------------------------------------------------------------
df["has_energy_cert"] = df["energy_performance_category"].notna().astype(int)
df["has_ghg_value"]   = df["ghg_value"].notna().astype(int)
df = df.drop(columns=["energy_performance_category", "energy_performance_value",
                       "ghg_value", "ghg_category"])


# ------------------------------------------------------------------
# 6. CODIFICACIÓN DE DEPARTAMENTO
# Los códigos postales franceses tienen 5 dígitos; los primeros 2 indican
# el departamento. Se aplica zfill(5) para normalizar códigos cortos
# (ej. "9700" → "09700" → dept "09") y se agrupa Córcega (2A/2B → 20).
# Se descartan departamentos con menos de 30 muestras.
# ------------------------------------------------------------------
df["dept"] = df["postal_code"].astype(str).str.zfill(5).str[:2]
df["dept"] = df["dept"].replace({"2A": "20", "2B": "20"})

depts_validos = df["dept"].value_counts()[lambda x: x >= 30].index
df = df[df["dept"].isin(depts_validos)].copy()

dept_to_int = {d: i for i, d in enumerate(sorted(df["dept"].unique()))}
df["dept_enc"] = df["dept"].map(dept_to_int)

print(f"Departamentos codificados: {len(dept_to_int)}")
print(f"Filas tras filtro de departamento: {len(df)}\n")


# ------------------------------------------------------------------
# 7. ELIMINAR COLUMNAS NO NECESARIAS
# ------------------------------------------------------------------
df = df.drop(columns=["id_annonce", "city", "postal_code", "property_type", "dept"])


# ------------------------------------------------------------------
# 8. ELIMINAR FILAS CON NULOS
# ------------------------------------------------------------------
n_antes = len(df)
df = df.dropna()
print(f"Filas eliminadas por nulos: {n_antes - len(df)}")
print(f"Filas finales: {len(df)}\n")


# ------------------------------------------------------------------
# 9. DISTANCIAS HAVERSINE A PRINCIPALES CIUDADES
# Añade la distancia en km a cada ciudad y la distancia mínima.
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
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

lat = df["approximate_latitude"].values
lon = df["approximate_longitude"].values
for nombre, (clat, clon) in CIUDADES.items():
    df[nombre] = haversine_vec(lat, lon, clat, clon)
df["dist_min_ciudad"] = df[list(CIUDADES)].min(axis=1)


# ------------------------------------------------------------------
# 10. FEATURE ENGINEERING
# Log del tamaño para reducir skewness; codificación ordinal del grupo.
# ------------------------------------------------------------------
df["log_size"] = np.log1p(df["size"])
df = df.drop(columns=["size"])

grupo_to_int = {"appartement": 0, "maison": 1, "lujo": 2}
df["property_group_enc"] = df["property_group"].map(grupo_to_int)
df = df.drop(columns=["property_group"])

print(f"Features: {[c for c in df.columns if c != 'price']}\n")


# ------------------------------------------------------------------
# 11. SPLIT TRAIN / TEST
# ------------------------------------------------------------------
X = df.drop(columns=["price"])
y_log = np.log1p(df["price"].values)

# Normalizar nombres de columnas (eliminar caracteres no-ASCII)
X.columns = (X.columns
               .str.normalize("NFKD")
               .str.encode("ascii", errors="ignore")
               .str.decode("ascii"))

X_train, X_test, y_train, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test = np.expm1(y_test_log)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")


# ------------------------------------------------------------------
# 12. ENTRENAMIENTO DEL MODELO XGBOOST
# ------------------------------------------------------------------
modelo = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="mae",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=1,
    tree_method="hist",
    verbosity=1,
)

print("=== ENTRENANDO XGBOOST ===")
modelo.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test_log)],
    verbose=100,
)
print(f"\nMejor iteración: {modelo.best_iteration}\n")


# ------------------------------------------------------------------
# 13. EVALUACIÓN
# ------------------------------------------------------------------
y_pred_log = modelo.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_pred     = np.clip(y_pred, df["price"].min(), df["price"].max())

mae  = mean_absolute_error(y_test, y_pred)
rsme = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("=== RESULTADOS ===")
print(f"MAE  : {mae:,.0f} €")
print(f"RMSE : {rsme:,.0f} €")
print(f"MAPE : {mape:.1f}%")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------------
# 14. GRÁFICAS DE DIAGNÓSTICO
# Genera scatter real vs predicho, análisis de residuos e importancia
# de variables. Guarda la imagen en outputs/.
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("XGBoost — Predicción de Precios de Viviendas", fontsize=14)

fmt = mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")

ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.25, s=8, color="steelblue")
max_val = float(max(y_test.max(), y_pred.max()))
ax1.plot([0, max_val], [0, max_val], "r--", lw=1.5)
ax1.set_xlabel("Precio real (€)")
ax1.set_ylabel("Precio predicho (€)")
ax1.set_title("Precios Reales vs Predichos")
ax1.xaxis.set_major_formatter(fmt)
ax1.yaxis.set_major_formatter(fmt)

ax2 = axes[1]
residuos = y_test - y_pred
ax2.scatter(y_pred, residuos, alpha=0.25, s=8, color="coral")
ax2.axhline(0, color="black", lw=1.2, linestyle="--")
ax2.set_xlabel("Precio predicho (€)")
ax2.set_ylabel("Residuo (€)")
ax2.set_title("Análisis de Residuos")
ax2.xaxis.set_major_formatter(fmt)
ax2.yaxis.set_major_formatter(fmt)

ax3 = axes[2]
importancias = (pd.Series(modelo.feature_importances_, index=X_train.columns)
                  .sort_values()
                  .tail(20))
importancias.plot(kind="barh", ax=ax3, color="steelblue")
ax3.set_title("Top 20 Variables más Importantes")
ax3.set_xlabel("Importancia (gain)")

fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=80)
buf.seek(0)
img = Image.open(buf)
img.save(os.path.join(OUT_DIR, "resultados_xgb.png"))
buf.close()
plt.close(fig)
print(f"Gráfica guardada en '{OUT_DIR}/resultados_xgb.png'")


# ------------------------------------------------------------------
# 15. DIAGNÓSTICOS POR GRUPO (plots_trees)
# ------------------------------------------------------------------
GROUP_MAP = {0: "appartement", 1: "maison", 2: "lujo"}

plots_trees(
    y_true        = y_test,
    y_pred        = y_pred,
    nombre        = "XGBoost",
    out_dir       = OUT_DIR,
    modelo        = modelo,
    feature_names = list(X_train.columns),
    group_labels  = X_test["property_group_enc"].values,
    group_map     = GROUP_MAP,
)


# ------------------------------------------------------------------
# 16. GUARDAR MODELO Y MÉTRICAS
# ------------------------------------------------------------------
joblib.dump(modelo,                  "models_pkl/xgboost.pkl")
joblib.dump(list(X_train.columns),   "models_pkl/xgboost_feature_cols.pkl")

pd.DataFrame([{
    "RMSE": np.sqrt(np.mean((y_test - y_pred) ** 2)),
    "MAE":  mae,
    "R2":   r2,
}]).to_csv(os.path.join(OUT_DIR, "metrics_xgb.csv"), index=False)

pd.DataFrame({
    "y_real": y_test,
    "y_pred": y_pred,
}).to_csv(os.path.join(OUT_DIR, "predictions_xgb.csv"), index=False)

print("Modelo, métricas y predicciones guardados correctamente.")