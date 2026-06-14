# =============================================================
#  PREDICCIÓN DE PRECIOS DE VIVIENDAS - Red Neuronal Multicapa
#  (Multilayer Perceptron - MLP) 
# =============================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from diagnostics import mae_por_decil, plots_mlp

ruta = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
outputs_dir = os.path.join(ruta, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

print(f"Ruta del proyecto: {ruta}\n")


# ------------------------------------------------------------------
# 1. CARGAR LOS DATOS
# ------------------------------------------------------------------
df = pd.read_csv(os.path.join(ruta, "Data/trainData.csv"))

print("=== DATOS CARGADOS ===")
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
print(f"Precio medio inicial: {df['price'].mean():,.0f} €\n")


# ------------------------------------------------------------------
# 2. AGRUPAR TIPOS DE VIVIENDA EN 3 CATEGORÍAS Y FILTRAR
#    Mapea los tipos originales a: appartement, maison, lujo.
#    Las filas con tipos no mapeados se eliminan.
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


# ------------------------------------------------------------------
# 3. ELIMINAR OUTLIERS DE PRECIO (percentil 99)
# ------------------------------------------------------------------
precio_p99 = df["price"].quantile(0.99)
df = df[df["price"] <= precio_p99].copy()
print(f"Precio máximo tras cap p99: {precio_p99:,.0f} €\n")


# ------------------------------------------------------------------
# 4. FILTRAR TAMAÑOS E PRECIO/M² IMPOSIBLES
# ------------------------------------------------------------------
df = df[(df["size"] >= 10) & (df["size"] <= 5000)].copy()
print(f"Filas tras filtro de size [10, 5000 m²]: {len(df)}\n")

df["price_m2_temp"] = df["price"] / df["size"]
n_antes = len(df)
df = df[(df["price_m2_temp"] >= 300) & (df["price_m2_temp"] <= 20000)].copy()
df = df.drop(columns=["price_m2_temp"])
print(f"Filas tras filtro precio/m² [300–20.000]: {len(df)} "
      f"(eliminadas {n_antes - len(df)})\n")


# ------------------------------------------------------------------
# 5. CODIFICACIÓN POR DEPARTAMENTO (2 primeros dígitos del CP)
#    zfill(5) garantiza que códigos cortos (e.g. 9700 → 09700)
#    se mapeen al departamento correcto (09, no 97).
#    Córcega (2A/2B) se unifica como '20'.
#    Departamentos con menos de 30 anuncios se eliminan.
# ------------------------------------------------------------------
df["dept"] = df["postal_code"].astype(str).str.zfill(5).str[:2]
df["dept"] = df["dept"].replace({"2A": "20", "2B": "20"})

dept_counts = df["dept"].value_counts()
depts_validos = dept_counts[dept_counts >= 30].index
df = df[df["dept"].isin(depts_validos)].copy()

dept_to_int = {d: i for i, d in enumerate(sorted(df["dept"].unique()))}
df["dept_enc"] = df["dept"].map(dept_to_int)
df = df.drop(columns=["dept"])
print(f"Departamentos conservados: {df['dept_enc'].nunique()}\n")


# ------------------------------------------------------------------
# 6. FILTRAR ESCENARIOS IMPOSIBLES EN VARIABLES NUMÉRICAS
# ------------------------------------------------------------------
df = df[~(df["nb_bedrooms"] > df["nb_rooms"])].copy()
print(f"Filas tras eliminar nb_bedrooms > nb_rooms: {len(df)}\n")


# ------------------------------------------------------------------
# 7. INDICADOR BOOLEANO DE CERTIFICADO ENERGÉTICO
#    En lugar de imputar, se crean banderas binarias (has_energy_cert,
#    has_ghg_value). La ausencia del certificado suele indicar
#    inmuebles antiguos.
# ------------------------------------------------------------------
df["has_energy_cert"] = df["energy_performance_category"].notna().astype(int)
df["has_ghg_value"]   = df["ghg_value"].notna().astype(int)
df = df.drop(columns=[
    "energy_performance_category", "energy_performance_value",
    "ghg_value", "ghg_category",
])


# ------------------------------------------------------------------
# 8. ELIMINAR COLUMNAS CON >50% NULOS Y COLUMNAS NO NECESARIAS
# ------------------------------------------------------------------
COLS_ALTO_NULO = ["exposition", "floor", "land_size", "nb_bathrooms"]
df = df.drop(columns=COLS_ALTO_NULO)

df = df.drop(columns=["id_annonce", "city", "postal_code", "property_type"])


# ------------------------------------------------------------------
# 9. ELIMINAR FILAS CON NULOS RESTANTES (sin imputación)
# ------------------------------------------------------------------
n_antes = len(df)
df = df.dropna()
print(f"Filas eliminadas por nulos restantes: {n_antes - len(df)}")
print(f"Filas finales limpias: {len(df)}")
print(f"Nulos totales: {df.isnull().sum().sum()}\n")


# ------------------------------------------------------------------
# 10. DISTANCIA A CIUDADES PRINCIPALES (km, haversine)
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

for nombre, (lat, lon) in CIUDADES.items():
    df[nombre] = df.apply(
        lambda row, la=lat, lo=lon: haversine(
            row["approximate_latitude"], row["approximate_longitude"], la, lo
        ),
        axis=1,
    )
df["dist_min_ciudad"] = df[[*CIUDADES]].min(axis=1)


# ------------------------------------------------------------------
# 11. TRANSFORMACIONES DE VARIABLES
#     log1p en size; coordenadas y size original se eliminan tras esto.
# ------------------------------------------------------------------
df["log_size"] = np.log1p(df["size"])
df = df.drop(columns=["approximate_latitude", "approximate_longitude", "size"])


# ------------------------------------------------------------------
# 12. ONE-HOT ENCODING
# ------------------------------------------------------------------
df = pd.get_dummies(df, columns=["property_group", "dept_enc"])


# ------------------------------------------------------------------
# 13. EXPORTAR DATOS PROCESADOS
# ------------------------------------------------------------------
df.to_csv(os.path.join(ruta, "Data/datos_procesados_v2.csv"), index=False)

print("=== VARIABLES USADAS EN EL MODELO ===")
feature_cols = [c for c in df.columns if c != "price"]
print(feature_cols)
print(f"\nTotal variables: {len(feature_cols)}\n")
print("Datos exportados a Data/datos_procesados_v2.csv\n")


# ------------------------------------------------------------------
# 14. SEPARAR FEATURES Y OBJETIVO / TRAIN-TEST SPLIT
# ------------------------------------------------------------------
X_proc = df.drop(columns=["price"])
y_proc = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_proc, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")


# ------------------------------------------------------------------
# 15. ESCALAR FEATURES Y PRECIO (log + StandardScaler)
# ------------------------------------------------------------------
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

y_train_log    = np.log1p(y_train)
scaler_y       = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1)).ravel()


# ------------------------------------------------------------------
# 16. ENTRENAR LA RED NEURONAL
# ------------------------------------------------------------------
modelo = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.0001,
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=80,
    tol=1e-5,
    validation_fraction=0.1,
    alpha=0.001,
    random_state=42,
    verbose=True,
)

print("=== ENTRENANDO LA RED NEURONAL ===")
modelo.fit(X_train_sc, y_train_scaled)
print()


# ------------------------------------------------------------------
# 17. EVALUAR EL MODELO
# ------------------------------------------------------------------
y_pred_scaled = modelo.predict(X_test_sc)
y_pred_log    = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_pred        = np.expm1(y_pred_log)
y_pred        = np.clip(y_pred, df["price"].min(), df["price"].max())

mae = mean_absolute_error(y_test, y_pred)
rsme = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2  = r2_score(y_test, y_pred)

print("=== RESULTADOS ===")
print(f"MAE  (Error Absoluto Medio) : {mae:,.0f} €")
print(f"  → En promedio, nos equivocamos en {mae:,.0f} € por vivienda")
print(f"RMSE (Error Cuadrático Medio) : {rsme:,.0f} €")
print(f"R²   (Coeficiente de determinación): {r2:.4f}")
print(f"  → El modelo explica el {r2*100:.1f}% de la variación en precios")


# ------------------------------------------------------------------
# 18. GRÁFICAS Y EXPORTACIÓN DE RESULTADOS
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Red Neuronal Multicapa - Predicción de Precios (v2)", fontsize=14)

ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
max_val = max(y_test.max(), y_pred.max())
ax1.plot([0, max_val], [0, max_val], "r--", label="Predicción perfecta")
ax1.set_xlabel("Precio real (€)")
ax1.set_ylabel("Precio predicho (€)")
ax1.set_title("Precios Reales vs Predichos")
ax1.legend()

ax2 = axes[1]
ax2.plot(modelo.loss_curve_, label="Train loss")
if modelo.validation_scores_ is not None:
    val_loss = [1 - s for s in modelo.validation_scores_]
    ax2.plot(val_loss, label="Validation loss")
ax2.set_xlabel("Época")
ax2.set_ylabel("Pérdida")
ax2.set_title("Curva de Aprendizaje")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "resultados_mlp_v2.png"), dpi=150)
plt.show()
print("\nGráfica guardada en outputs/resultados_mlp_v2.png")

joblib.dump(scaler_X,             os.path.join(outputs_dir, "mlp_scaler_X.pkl"))
joblib.dump(scaler_y,             os.path.join(outputs_dir, "mlp_scaler_y.pkl"))
joblib.dump(list(X_proc.columns), os.path.join(outputs_dir, "mlp_feature_cols.pkl"))

pd.DataFrame([{
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE":  mae,
    "R2":   r2,
}]).to_csv(os.path.join(outputs_dir, "metrics_mlp.csv"), index=False)

pd.DataFrame({
    "y_real": y_test,
    "y_pred": y_pred,
}).to_csv(os.path.join(outputs_dir, "predictions_mlp.csv"), index=False)

plots_mlp(
    y_test, y_pred,
    out_dir=outputs_dir,
    modelo=modelo,
    nombre="Perceptrón Multicapa (MLP)",
)
mae_por_decil(y_test, y_pred)