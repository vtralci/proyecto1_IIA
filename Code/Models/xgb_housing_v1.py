# =============================================================
#  PREDICCIÓN DE PRECIOS DE VIVIENDAS — XGBoost VERSION 1
# =============================================================
import joblib
import matplotlib
matplotlib.use("Agg")
import os
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import resample
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from diagnostics import mae_por_decil, plots_trees
import pickle


ruta = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------
# 1. CARGAR LOS DATOS
# ------------------------------------------------------------------
df = pd.read_csv(os.path.join(ruta, "Data/trainData.csv"))
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(f"Precio medio inicial: {df['price'].mean():,.0f} €\n")

# ------------------------------------------------------------------
# 2. AGRUPAR TIPOS DE VIVIENDA
# ------------------------------------------------------------------
PROPERTY_TYPE_MAP = {
    "appartement": ["appartement", "duplex", "loft"],
    "maison":      ["maison", "ferme", "chalet"],
    "lujo":        ["villa", "propriété"],
}
tipo_a_grupo = {t: g for g, ts in PROPERTY_TYPE_MAP.items() for t in ts}
df["property_group"] = df["property_type"].map(tipo_a_grupo)
df = df[df["property_group"].notna()].copy()

checkpoint = {}

# Tras paso 2
checkpoint["tras_property_group"] = len(df)

# Tras paso 3
checkpoint["tras_filtros_precio_size_bedrooms"] = len(df)

# Tras paso 4 (drop columnas alto nulo)
checkpoint["tras_drop_columnas"] = len(df)

# Tras dept filter
checkpoint["tras_dept_filter"] = len(df)

# Tras dropna
checkpoint["tras_dropna"] = len(df)

print(pd.Series(checkpoint).to_string())

# ------------------------------------------------------------------
# 3. FILTROS
# ------------------------------------------------------------------
print(f"1. tras property_group: {len(df)}")
precio_p99 = df["price"].quantile(0.99)
df = df[df["price"] <= precio_p99].copy()
print(f"2. tras precio p99: {len(df)}")
df = df[(df["size"] >= 10) & (df["size"] <= 5000)].copy()
print(f"3. tras size: {len(df)}")
df = df[~(df["nb_bedrooms"] > df["nb_rooms"])].copy()
print(f"4. tras bedrooms>rooms: {len(df)}")



# ------------------------------------------------------------------
# 4. COLUMNAS CON ALTO % DE NULOS
# ------------------------------------------------------------------
df = df.drop(columns=["exposition", "floor", "land_size", 'nb_bathrooms'])
print(f"5. tras drop columnas: {len(df)}")

# ------------------------------------------------------------------
# 5. CERTIFICADO ENERGÉTICO Y GHG
# ------------------------------------------------------------------
df["has_energy_cert"] = df["energy_performance_category"].notna().astype(int)
df["has_ghg_value"] = df["ghg_value"].notna().astype(int)
df = df.drop(columns=["energy_performance_category", "energy_performance_value", "ghg_value", "ghg_category"])

# ------------------------------------------------------------------
# 6. DEPARTAMENTO
# ------------------------------------------------------------------
df["dept"] = df["postal_code"].astype(str).str.zfill(5).str[:2]
df["dept"] = df["dept"].replace({"2A": "20", "2B": "20"})
depts_validos = df["dept"].value_counts()[lambda x: x >= 30].index
df = df[df["dept"].isin(depts_validos)].copy()
dept_to_int = {d: i for i, d in enumerate(sorted(df["dept"].unique()))}
df["dept_enc"] = df["dept"].map(dept_to_int)
# dept filter aquí
print(f"6. tras dept filter: {len(df)}")

# ------------------------------------------------------------------
# 7. ELIMINAR COLUMNAS NO NECESARIAS
# ------------------------------------------------------------------
df = df.drop(columns=["id_annonce", "city", "postal_code", "property_type", "dept"])

# ------------------------------------------------------------------
# 8. ELIMINAR NULOS
# ------------------------------------------------------------------
n_antes = len(df)
print(f"\nFilas antes de dropna: {len(df)}")

nulos = df.isnull().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)
print("\nColumnas con nulos:")
print(nulos.to_string())

# Cuántas filas se pierden por cada columna individualmente
print("\nFilas únicas eliminadas si solo se hiciera dropna en esa columna:")
for col in nulos.index:
    n = df[col].isnull().sum()
    print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")

# Overlap: filas con múltiples nulos
print(f"\nFilas con ≥1 nulo (total que dropna elimina): {df.isnull().any(axis=1).sum()}")
print(f"Filas con ≥2 nulos: {(df.isnull().sum(axis=1) >= 2).sum()}")

df = df.dropna()
print(f"Filas eliminadas por nulos: {n_antes - len(df)}")
print(f"Filas finales: {len(df)}\n")
print(f"7. tras dropna: {len(df)}")


# ------------------------------------------------------------------
# 9. HAVERSINE VECTORIZADO
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
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

lat = df["approximate_latitude"].values
lon = df["approximate_longitude"].values
for nombre, (clat, clon) in CIUDADES.items():
    df[nombre] = haversine_vec(lat, lon, clat, clon)
df["dist_min_ciudad"] = df[[*CIUDADES]].min(axis=1)

# ------------------------------------------------------------------
# 10. FEATURE ENGINEERING
# ------------------------------------------------------------------
df["log_size"] = np.log1p(df["size"])
df = df.drop(columns=["size"])

grupo_to_int = {"appartement": 0, "maison": 1, "lujo": 2}
df["property_group_enc"] = df["property_group"].map(grupo_to_int)
df = df.drop(columns=["property_group"])

print(f"Features: {[c for c in df.columns if c != 'price']}\n")

# ------------------------------------------------------------------
# 11. SPLIT
# ------------------------------------------------------------------
X = df.drop(columns=["price"])
y_log = np.log1p(df["price"].values)

# Limpiar nombres de columnas
X.columns = X.columns.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")

X_train, X_test, y_train, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test = np.expm1(y_test_log)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ------------------------------------------------------------------
# 12. MODELO
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
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("=== RESULTADOS ===")
print(f"MAE  : {mae:,.0f} €")
print(f"MAPE : {mape:.1f}%")
print(f"R²   : {r2:.4f}")

# ------------------------------------------------------------------
# 14. GRÁFICAS
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# GRÁFICAS — PIL (sin dependencias C problemáticas)
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
ax2.set_ylabel("Residuo €")
ax2.set_title("Análisis de Residuos")
ax2.xaxis.set_major_formatter(fmt)
ax2.yaxis.set_major_formatter(fmt)

ax3 = axes[2]
importancias = pd.Series(
    modelo.feature_importances_,
    index=X_train.columns
).sort_values().tail(20)
importancias.plot(kind="barh", ax=ax3, color="steelblue")
ax3.set_title("Top 20 Variables más Importantes")
ax3.set_xlabel("Importancia (gain)")

fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=80)
buf.seek(0)
img = Image.open(buf)
out_path = os.path.join(ruta, "resultados_xgb.png")
img.save(out_path)
buf.close()
plt.close(fig)
print(f"Gráfica guardada como '{out_path}'")

GROUP_MAP = {0: "appartement", 1: "maison", 2: "lujo"}

df_decil = plots_trees( 
    y_true        = y_test,
    y_pred        = y_pred,
    nombre        = "XGBoost",
    out_dir       = ruta,
    modelo        = modelo,
    feature_names = list(X_train.columns),
    group_labels  = X_test["property_group_enc"].values,
    group_map     = GROUP_MAP,
)

joblib.dump(modelo, 'models_pkl/xgboost.pkl')
joblib.dump(list(X_train.columns), 'models_pkl/xgboost_feature_cols.pkl')


import os
os.makedirs('outputs', exist_ok=True)
pd.DataFrame([{"RMSE": np.sqrt(np.mean((y_test - y_pred)**2)), "MAE": mae, "R2": r2}]).to_csv('outputs/metrics_xgb.csv', index=False)
pd.DataFrame({"y_real": y_test, "y_pred": y_pred}).to_csv('outputs/predictions_xgb.csv', index=False)