# =============================================================
#  PREDICCIÓN DE PRECIOS DE VIVIENDAS — Ridge
#  Median Target Encoding + Eliminación de outliers influyentes
# =============================================================

import os
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from diagnostics import mae_por_decil, plots_mlp

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# PALETA DE COLORES
# ------------------------------------------------------------------
C_BG     = "#F8F7F4"
C_TEXT   = "#2C2C2A"
C_MUTED  = "#5F5E5A"
C_PURPLE = "#534AB7"
C_TEAL   = "#0F6E56"
C_AMBER  = "#854F0B"
C_RED    = "#A32D2D"
C_BLUE   = "#185FA5"


def style_ax(ax, title, xlabel, ylabel):
    """Aplica el estilo visual uniforme a un eje de matplotlib."""
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values():
        sp.set_color("#D3D1C7")
        sp.set_linewidth(0.6)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=C_MUTED, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=8.5, color=C_MUTED, labelpad=4)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#D3D1C7")


MAX_SCATTER = 3_000

def subsample(n, size=MAX_SCATTER):
    """Devuelve índices ordenados de una muestra aleatoria de tamaño `size`."""
    if n > size:
        idx = np.random.choice(n, size=size, replace=False)
        idx.sort()
        return idx
    return np.arange(n)


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
df = df[(df["size"] >= 10) & (df["size"] <= 5_000)].copy()
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
# La columna dept se conserva como string hasta el target encoding (paso 12).
# ------------------------------------------------------------------
df["dept"] = df["postal_code"].astype(str).str.zfill(5).str[:2]
df["dept"] = df["dept"].replace({"2A": "20", "2B": "20"})

depts_validos = df["dept"].value_counts()[lambda x: x >= 30].index
df = df[df["dept"].isin(depts_validos)].copy()

print(f"Departamentos conservados: {df['dept'].nunique()}")
print(f"Filas tras filtro de departamento: {len(df)}\n")


# ------------------------------------------------------------------
# 7. ELIMINAR COLUMNAS NO NECESARIAS
# ------------------------------------------------------------------
df = df.drop(columns=["id_annonce", "city", "postal_code", "property_type"])


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
    """Distancia Haversine vectorizada entre un array de puntos y un punto fijo (km)."""
    R = 6_371
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
# Log del tamaño para reducir skewness.
# ------------------------------------------------------------------
df["log_size"] = np.log1p(df["size"])
df = df.drop(columns=["size"])

print(f"Features: {[c for c in df.columns if c != 'price']}\n")


# ------------------------------------------------------------------
# 11. SPLIT TRAIN / TEST
# El split se realiza ANTES del encoding para evitar data leakage.
# La variable objetivo se transforma en log para estabilizar varianza.
# ------------------------------------------------------------------
X = df.drop(columns=["price"])
y_log = np.log1p(df["price"].values)

X_train, X_test, y_train, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test = np.expm1(y_test_log)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")


# ------------------------------------------------------------------
# 12. MEDIAN TARGET ENCODING
# Se calcula la mediana del precio POR GRUPO solo en train y se aplica
# a test para evitar leakage. Grupos sin mapping reciben la mediana global.
# Ridge requiere variables numéricas, por lo que dept y property_group
# se sustituyen por su mediana de precio en lugar de un label entero.
# ------------------------------------------------------------------
train_prices  = pd.Series(np.expm1(y_train), index=X_train.index)
global_median = float(train_prices.median())

dept_median_map = (
    X_train.assign(price=train_prices)
           .groupby("dept")["price"].median()
)
X_train["dept_enc"] = X_train["dept"].map(dept_median_map).fillna(global_median)
X_test["dept_enc"]  = X_test["dept"].map(dept_median_map).fillna(global_median)

group_median_map = (
    X_train.assign(price=train_prices)
           .groupby("property_group")["price"].median()
)
X_train["property_group_enc"] = X_train["property_group"].map(group_median_map).fillna(global_median)
X_test["property_group_enc"]  = X_test["property_group"].map(group_median_map).fillna(global_median)

X_train = X_train.drop(columns=["dept", "property_group"])
X_test  = X_test.drop(columns=["dept", "property_group"])

# Normalizar nombres de columnas a ASCII
def clean_cols(df_):
    df_.columns = (df_.columns
                       .str.normalize("NFKD")
                       .str.encode("ascii", errors="ignore")
                       .str.decode("ascii"))
    return df_

X_train = clean_cols(X_train)
X_test  = clean_cols(X_test)

print(f"Features finales ({X_train.shape[1]}): {list(X_train.columns)}\n")


# ------------------------------------------------------------------
# 13. ESCALADO DE FEATURES
# StandardScaler ajustado solo en train y aplicado a test.
# Ridge es sensible a la escala — este paso es obligatorio.
# ------------------------------------------------------------------
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)


# ------------------------------------------------------------------
# 14. MODELO RIDGE INICIAL
# RidgeCV selecciona el alpha óptimo mediante validación cruzada 5-fold.
# ------------------------------------------------------------------
alphas = np.logspace(-1, 3, 50)
modelo = RidgeCV(alphas=alphas, cv=5)

print("=== ENTRENANDO RIDGE (modelo inicial) ===")
modelo.fit(X_train_sc, y_train)
print(f"Alpha óptimo: {modelo.alpha_:.4f}\n")


# ------------------------------------------------------------------
# 15. DETECCIÓN DE OUTLIERS INFLUYENTES — Cook's Distance en train
# El leverage se calcula eficientemente via SVD para evitar construir
# la matriz hat completa (que sería O(n²) en memoria).
# ------------------------------------------------------------------
print("=== DETECTANDO OUTLIERS INFLUYENTES (Cook's Distance) ===")
U, s, Vt       = np.linalg.svd(X_train_sc, full_matrices=False)
hat_diag_train = np.clip(np.sum(U ** 2, axis=1), 1e-8, 1 - 1e-8)

y_pred_train = modelo.predict(X_train_sc)
resid_train  = y_train - y_pred_train
mse_tr       = np.mean(resid_train ** 2)
std_resid_tr = resid_train / np.sqrt(mse_tr)
p_tr         = X_train_sc.shape[1]

cooks_train    = (std_resid_tr ** 2 / p_tr) * (hat_diag_train / (1 - hat_diag_train) ** 2)
cooks_thresh   = 4 / len(X_train_sc)
mask_clean     = cooks_train <= cooks_thresh
n_outliers     = int((~mask_clean).sum())

print(f"Umbral Cook's D (4/n): {cooks_thresh:.6f}")
print(f"Outliers influyentes : {n_outliers} ({n_outliers / len(X_train_sc) * 100:.1f}%)")
print(f"Filas conservadas    : {mask_clean.sum()}\n")


# ------------------------------------------------------------------
# 16. REENTRENAMIENTO SIN OUTLIERS INFLUYENTES
# ------------------------------------------------------------------
X_train_clean = X_train_sc[mask_clean]
y_train_clean = y_train[mask_clean]

modelo_clean = RidgeCV(alphas=alphas, cv=5)
print("=== ENTRENANDO RIDGE (sin outliers) ===")
modelo_clean.fit(X_train_clean, y_train_clean)
print(f"Alpha óptimo (clean): {modelo_clean.alpha_:.4f}\n")


# ------------------------------------------------------------------
# 17. EVALUACIÓN Y COMPARATIVA DE MODELOS
# ------------------------------------------------------------------
def evaluar(nombre, mdl, X_sc, y_real):
    """Evalúa un modelo en espacio original (€) e imprime métricas."""
    y_pred_log_ = mdl.predict(X_sc)
    y_pred_     = np.expm1(y_pred_log_)
    y_pred_     = np.clip(y_pred_, df["price"].min(), df["price"].max())
    mae_  = mean_absolute_error(y_real, y_pred_)
    rmse_ = np.sqrt(mean_squared_error(y_real, y_pred_))
    r2_   = r2_score(y_real, y_pred_)
    mape_ = np.mean(np.abs((y_real - y_pred_) / y_real)) * 100
    print(f"=== {nombre} ===")
    print(f"  MAE  : {mae_:,.0f} €")
    print(f"  RMSE : {rmse_:,.0f} €")
    print(f"  MAPE : {mape_:.1f}%")
    print(f"  R²   : {r2_:.4f}\n")
    return y_pred_log_, y_pred_

y_pred_log_orig,  y_pred_orig  = evaluar("MODELO ORIGINAL",     modelo,       X_test_sc, y_test)
y_pred_log_clean, y_pred_clean = evaluar("MODELO SIN OUTLIERS", modelo_clean, X_test_sc, y_test)


# ------------------------------------------------------------------
# 18. GUARDAR MODELO, PREDICCIONES Y MÉTRICAS
# ------------------------------------------------------------------
joblib.dump(modelo_clean, os.path.join(OUT_DIR, "linear_regression.pkl"))

pd.DataFrame({
    "y_real":        y_test,
    "y_pred_orig":   y_pred_orig,
    "y_pred_clean":  y_pred_clean,
    "residuo_orig":  y_test - y_pred_orig,
    "residuo_clean": y_test - y_pred_clean,
}).to_csv(os.path.join(OUT_DIR, "resultados_ridge.csv"), index=False)

pd.DataFrame([
    {"modelo": "Ridge_orig",  "MAE": mean_absolute_error(y_test, y_pred_orig),
     "MAPE": np.mean(np.abs((y_test - y_pred_orig)  / y_test)) * 100,
     "R2": r2_score(y_test, y_pred_orig),  "alpha": modelo.alpha_,
     "RMSE": np.sqrt(np.mean((y_test - y_pred_orig) ** 2))},
    {"modelo": "Ridge_clean", "MAE": mean_absolute_error(y_test, y_pred_clean),
     "MAPE": np.mean(np.abs((y_test - y_pred_clean) / y_test)) * 100,
     "R2": r2_score(y_test, y_pred_clean), "alpha": modelo_clean.alpha_,
     "RMSE": np.sqrt(np.mean((y_test - y_pred_clean) ** 2))},
]).to_csv(os.path.join(OUT_DIR, "metricas_ridge.csv"), index=False)

print(f"Resultados guardados en: {OUT_DIR}\n")


# ------------------------------------------------------------------
# 19. COEFICIENTES DEL MODELO LIMPIO
# Se muestran los 10 coeficientes con mayor valor absoluto.
# ------------------------------------------------------------------
coefs = pd.Series(np.abs(modelo_clean.coef_), index=X_train.columns).sort_values(ascending=False)

top10 = (
    pd.Series(modelo_clean.coef_, index=X_train.columns)
      .to_frame("coef")
      .assign(abs_coef=lambda d: d["coef"].abs())
      .sort_values("abs_coef", ascending=False)
      .head(10)
      .drop(columns="abs_coef")
)
print(top10.to_string(), "\n")


# ------------------------------------------------------------------
# 20. RESUMEN ESTADÍSTICO (aproximado)
# Nota: Ridge penaliza los coeficientes → SE, t y p-valores son
# aproximaciones basadas en OLS y deben interpretarse con cautela.
# ------------------------------------------------------------------
n = X_train_clean.shape[0]
p = X_train_clean.shape[1]

resid_clean_tr = y_train_clean - modelo_clean.predict(X_train_clean)
mse_clean_tr   = np.sum(resid_clean_tr ** 2) / (n - p - 1)
XtX_inv        = np.linalg.inv(X_train_clean.T @ X_train_clean)
se_betas       = np.sqrt(np.diag(XtX_inv * mse_clean_tr))
t_values       = modelo_clean.coef_ / se_betas
p_values       = 2 * stats.t.sf(np.abs(t_values), df=n - p - 1)
ci_low         = modelo_clean.coef_ - 1.96 * se_betas
ci_high        = modelo_clean.coef_ + 1.96 * se_betas

def sig_stars(p_):
    """Devuelve asteriscos de significancia estadística al estilo R."""
    if p_ < 0.001: return "***"
    if p_ < 0.01:  return "** "
    if p_ < 0.05:  return "*  "
    if p_ < 0.1:   return ".  "
    return "   "

hdr = (f"\n{'Variable':<35} {'Estimate':>10} {'Std. Error':>11} "
       f"{'t value':>9} {'Pr(>|t|)':>10} {'CI 95% Low':>11} {'CI 95% High':>12}  Sig.")
print("=" * 78)
print("RESUMEN ESTADÍSTICO — Ridge (modelo sin outliers)")
print("=" * 78)
print(hdr)
print("-" * len(hdr))

se_int      = np.sqrt(mse_clean_tr / n)
t_int       = modelo_clean.intercept_ / se_int
p_int       = 2 * stats.t.sf(abs(t_int), df=n - p - 1)
ci_int_low  = modelo_clean.intercept_ - 1.96 * se_int
ci_int_high = modelo_clean.intercept_ + 1.96 * se_int
print(f"{'(Intercept)':<35} {modelo_clean.intercept_:>10.6f} {se_int:>11.6f} "
      f"{t_int:>9.3f} {p_int:>10.4e} {ci_int_low:>11.6f} {ci_int_high:>12.6f}  {sig_stars(p_int)}")

for feat in coefs.index:
    i = list(X_train.columns).index(feat)
    print(f"{feat:<35} {modelo_clean.coef_[i]:>10.6f} {se_betas[i]:>11.6f} "
          f"{t_values[i]:>9.3f} {p_values[i]:>10.4e} "
          f"{ci_low[i]:>11.6f} {ci_high[i]:>12.6f}  {sig_stars(p_values[i])}")

print("-" * len(hdr))
print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
print("⚠  SE, t y p-valores son aproximados (Ridge penaliza coeficientes).\n")

ss_res = np.sum(resid_clean_tr ** 2)
ss_tot = np.sum((y_train_clean - y_train_clean.mean()) ** 2)
r2_tr  = 1 - ss_res / ss_tot
r2_adj = 1 - (1 - r2_tr) * (n - 1) / (n - p - 1)
f_stat = (r2_tr / p) / ((1 - r2_tr) / (n - p - 1))
f_pval = stats.f.sf(f_stat, dfn=p, dfd=n - p - 1)

print(f"Residual std error : {np.sqrt(mse_clean_tr):.6f}  (df = {n - p - 1})")
print(f"Multiple R²        : {r2_tr:.6f}")
print(f"Adjusted R²        : {r2_adj:.6f}")
print(f"F-statistic        : {f_stat:,.2f}  (df1={p}, df2={n-p-1})")
print(f"p-value (F-test)   : {f_pval:.4e}")
print("=" * 78 + "\n")


# ------------------------------------------------------------------
# 21. DIAGNÓSTICO GRÁFICO — modelo limpio (residuos en log-space)
# 7 subplots: residuos vs fitted, Q-Q, scale-location, distribución,
# Cook's D, VIF, residuos vs leverage.
# ------------------------------------------------------------------

# Residuos y métricas de diagnóstico en test
resid_test = y_test_log - y_pred_log_clean
fitted_log = y_pred_log_clean
mse_test_  = np.mean(resid_test ** 2)
std_resid  = resid_test / np.sqrt(mse_test_)
sqrt_abs   = np.sqrt(np.abs(std_resid))

# Leverage en test vía SVD
U_t, s_t, Vt_t = np.linalg.svd(X_test_sc, full_matrices=False)
hat_test        = np.clip(np.sum(U_t ** 2, axis=1), 1e-8, 1 - 1e-8)

# Cook's Distance en test
n_t          = len(resid_test)
p_t          = X_test_sc.shape[1]
cooks_test   = (std_resid ** 2 / p_t) * (hat_test / (1 - hat_test) ** 2)
cooks_thresh_t = 4 / n_t

# VIF calculado sobre train limpio
vif_values = np.zeros(p_t)
for i in range(p_t):
    y_i  = X_train_clean[:, i]
    X_i  = np.delete(X_train_clean, i, axis=1)
    beta = np.linalg.lstsq(X_i, y_i, rcond=None)[0]
    ss_r = np.sum((y_i - X_i @ beta) ** 2)
    ss_t_vif = np.sum((y_i - y_i.mean()) ** 2)
    r2_i = 1 - ss_r / ss_t_vif if ss_t_vif > 0 else 0
    vif_values[i] = 1 / (1 - r2_i) if r2_i < 1 else np.inf
vif_series = pd.Series(vif_values, index=X_train.columns).sort_values(ascending=False)

# Q-Q sobre submuestra
si             = subsample(n_t)
osm, osr       = stats.probplot(resid_test[si], dist="norm", fit=False)
sw_stat, sw_p  = stats.shapiro(resid_test[si[:min(2_000, len(si))]])
n_influyentes  = int(np.sum(cooks_test > cooks_thresh_t))

# Construcción de la figura
fig = plt.figure(figsize=(18, 14), facecolor=C_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.52, wspace=0.38,
                        top=0.90, bottom=0.06, left=0.07, right=0.97)
r2_test_log = r2_score(y_test_log, y_pred_log_clean)

fig.text(0.5, 0.955,
         "Comprobación de supuestos — Ridge (sin outliers, log-space residuals)",
         ha="center", fontsize=14, fontweight="bold", color=C_TEXT)
fig.text(0.5, 0.935,
         f"n_test={n_t:,}  |  n_outliers_eliminados={n_outliers}  |  "
         f"alpha={modelo_clean.alpha_:.4f}  |  R²_test={r2_test_log:.4f}",
         ha="center", fontsize=9, color=C_MUTED)

# Subplot 1: Residuos vs Fitted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(fitted_log[si], resid_test[si], s=5, alpha=0.3, c=C_PURPLE, edgecolors="none")
ax1.axhline(0, color=C_RED, linewidth=1.2, linestyle="--")
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(resid_test[si], fitted_log[si], frac=0.3)
    ax1.plot(lw[:, 0], lw[:, 1], color=C_AMBER, linewidth=1.6, label="LOWESS")
    ax1.legend(fontsize=7.5, framealpha=0.5)
except ImportError:
    pass
style_ax(ax1, "1. Residuos vs Fitted", "Fitted values (log)", "Residuals")

# Subplot 2: Q-Q Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(osm, osr, s=5, alpha=0.3, c=C_TEAL, edgecolors="none")
slope_qq, int_qq, *_ = stats.linregress(osm, osr)
line_x = np.array([osm[0], osm[-1]])
ax2.plot(line_x, slope_qq * line_x + int_qq, color=C_RED,
         linewidth=1.3, linestyle="--", label="Normal ref.")
ax2.text(0.04, 0.94, f"Shapiro-Wilk W={sw_stat:.4f}\np={sw_p:.2e}",
         transform=ax2.transAxes, fontsize=7.5, color=C_TEXT, va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#D3D1C7"))
ax2.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax2, "2. Normal Q-Q", "Theoretical quantiles", "Sample quantiles")

# Subplot 3: Scale-Location
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(fitted_log[si], sqrt_abs[si], s=5, alpha=0.3, c=C_BLUE, edgecolors="none")
try:
    lw2 = lowess(sqrt_abs[si], fitted_log[si], frac=0.3)
    ax3.plot(lw2[:, 0], lw2[:, 1], color=C_AMBER, linewidth=1.6, label="LOWESS")
    ax3.legend(fontsize=7.5, framealpha=0.5)
except Exception:
    pass
style_ax(ax3, "3. Scale-Location", "Fitted values (log)", "√|Std. residuals|")

# Subplot 4: Distribución de residuos
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(resid_test, bins=60, color=C_PURPLE, edgecolor="none", alpha=0.7, density=True)
xr = np.linspace(resid_test.min(), resid_test.max(), 300)
ax4.plot(xr, stats.norm.pdf(xr, resid_test.mean(), resid_test.std()),
         color=C_RED, linewidth=1.5, label="Normal teórica")
ax4.axvline(0, color=C_MUTED, linewidth=0.8, linestyle=":")
ax4.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax4, "4. Distribución de residuos", "Residual", "Densidad")

# Subplot 5: Cook's Distance
ax5 = fig.add_subplot(gs[1, 1])
MAX_BARS = 2_000
if n_t > MAX_BARS:
    idx_bar    = np.random.choice(n_t, size=MAX_BARS, replace=False)
    idx_bar.sort()
    cooks_plot = cooks_test[idx_bar]
    x_plot     = idx_bar
else:
    cooks_plot = cooks_test
    x_plot     = np.arange(n_t)
colors_cook = [C_RED if c > cooks_thresh_t else C_MUTED for c in cooks_plot]
ax5.bar(x_plot, cooks_plot, color=colors_cook,
        width=max(1, n_t // MAX_BARS), alpha=0.7)
ax5.axhline(cooks_thresh_t, color=C_RED, linewidth=1.2, linestyle="--",
            label=f"4/n = {cooks_thresh_t:.4f}")
ax5.text(0.98, 0.95, f"Influyentes: {n_influyentes} ({n_influyentes/n_t*100:.1f}%)",
         transform=ax5.transAxes, fontsize=8, color=C_RED, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#D3D1C7"))
ax5.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax5, "5. Cook's Distance (sample)", "Observación (índice)", "Cook's D")

# Subplot 6: VIF (Variance Inflation Factor)
ax6 = fig.add_subplot(gs[1, 2])
colors_vif = [C_RED if v > 10 else (C_AMBER if v > 5 else C_TEAL) for v in vif_series.values]
ax6.barh(range(len(vif_series)), vif_series.values, color=colors_vif, alpha=0.85)
ax6.set_yticks(range(len(vif_series)))
ax6.set_yticklabels(vif_series.index, fontsize=7.5)
ax6.axvline(5,  color=C_AMBER, linewidth=1, linestyle="--", alpha=0.8, label="VIF=5")
ax6.axvline(10, color=C_RED,   linewidth=1, linestyle="--", alpha=0.8, label="VIF=10")
ax6.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax6, "6. VIF — Multicolinealidad", "VIF", "")

# Subplot 7: Residuos vs Leverage (ocupa toda la fila inferior)
ax7 = fig.add_subplot(gs[2, :])
point_colors = np.where(cooks_test[si] > cooks_thresh_t, C_RED, C_PURPLE)
ax7.scatter(hat_test[si], std_resid[si], s=7, alpha=0.3,
            c=point_colors, edgecolors="none")
ax7.axhline(0, color=C_MUTED, linewidth=0.8, linestyle=":")
for lv, col in [(2, C_AMBER), (-2, C_AMBER), (3, C_RED), (-3, C_RED)]:
    ax7.axhline(lv, color=col, linewidth=0.8, linestyle="--", alpha=0.7)
n_out2 = int(np.sum(np.abs(std_resid) > 2))
n_out3 = int(np.sum(np.abs(std_resid) > 3))
legend_elems = [
    mpatches.Patch(color=C_RED,    label=f"Influyentes (Cook>4/n): {n_influyentes}"),
    mpatches.Patch(color=C_PURPLE, label="Normales"),
    Line2D([0], [0], color=C_AMBER, linestyle="--", linewidth=0.9, label=f"|std resid|>2: {n_out2}"),
    Line2D([0], [0], color=C_RED,   linestyle="--", linewidth=0.9, label=f"|std resid|>3: {n_out3}"),
]
ax7.legend(handles=legend_elems, fontsize=8, framealpha=0.6, ncol=4)
style_ax(ax7, "7. Residuos vs Leverage", "Leverage (hat value)", "Std. residuals")

plt.savefig(os.path.join(OUT_DIR, "ridge_assumptions.png"),
            dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Diagnóstico guardado en: {OUT_DIR}/ridge_assumptions.png\n")


# ------------------------------------------------------------------
# 22. DIAGNÓSTICO TEXTUAL FINAL
# ------------------------------------------------------------------
print("=== DIAGNÓSTICO DE SUPUESTOS (modelo sin outliers) ===")
print(f"  Shapiro-Wilk (normalidad)  W={sw_stat:.4f}  p={sw_p:.2e}  "
      f"{'⚠ No normal' if sw_p < 0.05 else '✓ Normal'}")
print(f"  Outliers |std resid| > 2  : {n_out2} ({n_out2/n_t*100:.1f}%)")
print(f"  Outliers |std resid| > 3  : {n_out3} ({n_out3/n_t*100:.1f}%)")
print(f"  Influyentes (Cook > 4/n)  : {n_influyentes} ({n_influyentes/n_t*100:.1f}%)")
print(f"\n  VIF — top 5:")
for feat, vif in vif_series.head(5).items():
    flag = " ⚠ ALTO" if vif > 10 else (" · moderado" if vif > 5 else "")
    print(f"    {feat:<35} {vif:>7.2f}{flag}")


# ------------------------------------------------------------------
# 23. PLOTS ADICIONALES VÍA diagnostics.py
# ------------------------------------------------------------------
plots_mlp(y_test, y_pred_orig,  out_dir=OUT_DIR, modelo=modelo,
          nombre="Regresión Ridge")
mae_por_decil(y_test, y_pred_orig)

plots_mlp(y_test, y_pred_clean, out_dir=OUT_DIR, modelo=modelo_clean,
          nombre="Regresión Ridge (sin outliers)")
mae_por_decil(y_test, y_pred_clean)