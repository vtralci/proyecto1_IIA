# =============================================================
#  PREDICCIÓN DE PRECIOS DE VIVIENDAS — Ridge v3
#  Median Target Encoding + Eliminación de outliers influyentes
# =============================================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ruta = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------
# PALETA
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
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values():
        sp.set_color("#D3D1C7")
        sp.set_linewidth(0.6)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=C_MUTED, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=8.5, color=C_MUTED, labelpad=4)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#D3D1C7")

MAX_SCATTER = 3000

def subsample(n, size=MAX_SCATTER):
    if n > size:
        idx = np.random.choice(n, size=size, replace=False)
        idx.sort()
        return idx
    return np.arange(n)

# ------------------------------------------------------------------
# 1. CARGAR DATOS
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

# ------------------------------------------------------------------
# 3. FILTROS
# ------------------------------------------------------------------
precio_p99 = df["price"].quantile(0.99)
df = df[df["price"] <= precio_p99].copy()
df = df[(df["size"] >= 10) & (df["size"] <= 5000)].copy()
df = df[~(df["nb_bedrooms"] > df["nb_rooms"])].copy()
print(f"Filas tras filtros: {len(df)}")

# ------------------------------------------------------------------
# 4. COLUMNAS CON ALTO % DE NULOS
# ------------------------------------------------------------------
df = df.drop(columns=["exposition", "floor", "land_size",
                       "ghg_value", "ghg_category", "nb_bathrooms"])

# ------------------------------------------------------------------
# 5. CERTIFICADO ENERGÉTICO
# ------------------------------------------------------------------
df["has_energy_cert"] = df["energy_performance_category"].notna().astype(int)
df = df.drop(columns=["energy_performance_category", "energy_performance_value"])

# ------------------------------------------------------------------
# 6. DEPARTAMENTO — extraer y filtrar
# ------------------------------------------------------------------
df["dept"] = df["postal_code"].astype(str).str.zfill(5).str[:2]
df["dept"] = df["dept"].replace({"2A": "20", "2B": "20"})

dept_counts  = df["dept"].value_counts()
depts_validos = dept_counts[dept_counts >= 30].index
n_antes = len(df)
df = df[df["dept"].isin(depts_validos)].copy()
print(f"Departamentos conservados: {df['dept'].nunique()} "
      f"(eliminadas {n_antes - len(df)} filas)")

# ------------------------------------------------------------------
# 7. ELIMINAR COLUMNAS NO NECESARIAS
# ------------------------------------------------------------------
df = df.drop(columns=["id_annonce", "city", "postal_code", "property_type"])

# ------------------------------------------------------------------
# 8. ELIMINAR NULOS
# ------------------------------------------------------------------
n_antes = len(df)
df = df.dropna()
print(f"Filas eliminadas por nulos: {n_antes - len(df)}")
print(f"Filas finales: {len(df)}\n")

# ------------------------------------------------------------------
# 9. DISTANCIAS HAVERSINE
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
df["dist_min_ciudad"] = df[[*CIUDADES]].min(axis=1)

# ------------------------------------------------------------------
# 10. FEATURE ENGINEERING
# ------------------------------------------------------------------
df["log_size"] = np.log1p(df["size"])
df = df.drop(columns=["size"])

# ------------------------------------------------------------------
# 11. SPLIT — antes del encoding para evitar data leakage
# ------------------------------------------------------------------
X = df.drop(columns=["price"])
y_log = np.log1p(df["price"].values)

X_train, X_test, y_train, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test = np.expm1(y_test_log)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ------------------------------------------------------------------
# 12. MEDIAN TARGET ENCODING — calculado solo sobre train
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

# Limpiar nombres de columnas
def clean_cols(df_):
    df_.columns = (df_.columns
                   .str.normalize("NFKD")
                   .str.encode("ascii", errors="ignore")
                   .str.decode("ascii"))
    return df_

X_train = clean_cols(X_train)
X_test  = clean_cols(X_test)

print("Features finales:", list(X_train.columns))
print(f"Total features: {X_train.shape[1]}\n")

# ------------------------------------------------------------------
# 13. ESCALADO
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ------------------------------------------------------------------
# 14. MODELO INICIAL
# ------------------------------------------------------------------
alphas = np.logspace(-1, 3, 50)
modelo = RidgeCV(alphas=alphas, cv=5)

print("=== ENTRENANDO RIDGE (modelo inicial) ===")
modelo.fit(X_train_sc, y_train)
print(f"Alpha óptimo: {modelo.alpha_:.4f}\n")

# ------------------------------------------------------------------
# 15. DETECCIÓN DE OUTLIERS INFLUYENTES — Cook's Distance en train
#     Leverage via SVD (eficiente, no construye la matriz hat completa)
# ------------------------------------------------------------------
print("=== DETECTANDO OUTLIERS INFLUYENTES (Cook's Distance) ===")

U, s, Vt = np.linalg.svd(X_train_sc, full_matrices=False)
hat_diag_train = np.sum(U ** 2, axis=1)          # diagonal de H = UU^T
hat_diag_train = np.clip(hat_diag_train, 1e-8, 1 - 1e-8)

y_pred_train   = modelo.predict(X_train_sc)
resid_train    = y_train - y_pred_train
mse_tr         = np.mean(resid_train ** 2)
std_resid_tr   = resid_train / np.sqrt(mse_tr)
p_tr           = X_train_sc.shape[1]

cooks_train    = (std_resid_tr ** 2 / p_tr) * (hat_diag_train / (1 - hat_diag_train) ** 2)
cooks_thresh_tr = 4 / len(X_train_sc)
mask_clean     = cooks_train <= cooks_thresh_tr

n_outliers = int((~mask_clean).sum())
print(f"Umbral Cook's D (4/n): {cooks_thresh_tr:.6f}")
print(f"Outliers influyentes : {n_outliers} ({n_outliers / len(X_train_sc) * 100:.1f}%)")
print(f"Filas conservadas    : {mask_clean.sum()}\n")

# ------------------------------------------------------------------
# 16. REENTRENAR SIN OUTLIERS
# ------------------------------------------------------------------
X_train_clean = X_train_sc[mask_clean]
y_train_clean = y_train[mask_clean]

modelo_clean = RidgeCV(alphas=alphas, cv=5)
print("=== ENTRENANDO RIDGE (sin outliers) ===")
modelo_clean.fit(X_train_clean, y_train_clean)
print(f"Alpha óptimo (clean): {modelo_clean.alpha_:.4f}\n")

# ------------------------------------------------------------------
# 17. EVALUACIÓN Y COMPARATIVA
# ------------------------------------------------------------------
def evaluar(nombre, mdl, X_sc, y_real, y_real_log, y_pred_log_fn=None):
    y_pred_log_ = mdl.predict(X_sc)
    y_pred_     = np.expm1(y_pred_log_)
    y_pred_     = np.clip(y_pred_, df["price"].min(), df["price"].max())
    mae_  = mean_absolute_error(y_real, y_pred_)
    r2_   = r2_score(y_real, y_pred_)
    mape_ = np.mean(np.abs((y_real - y_pred_) / y_real)) * 100
    print(f"=== {nombre} ===")
    print(f"  MAE  : {mae_:,.0f} €")
    print(f"  MAPE : {mape_:.1f}%")
    print(f"  R²   : {r2_:.4f}\n")
    return y_pred_log_, y_pred_

y_pred_log_orig,  y_pred_orig  = evaluar("MODELO ORIGINAL",     modelo,       X_test_sc, y_test, y_test_log)
y_pred_log_clean, y_pred_clean = evaluar("MODELO SIN OUTLIERS", modelo_clean, X_test_sc, y_test, y_test_log)

# ------------------------------------------------------------------
# 18. GUARDAR MODELO LIMPIO Y RESULTADOS
# ------------------------------------------------------------------
joblib.dump(modelo_clean, "models_pkl/linear_regression.pkl")

pd.DataFrame({
    "y_real": y_test,
    "y_pred_orig":  y_pred_orig,
    "y_pred_clean": y_pred_clean,
    "residuo_orig":  y_test - y_pred_orig,
    "residuo_clean": y_test - y_pred_clean,
}).to_csv(os.path.join(ruta, "resultados_ridge.csv"), index=False)

pd.DataFrame([
    {"modelo": "Ridge_orig",  "MAE": mean_absolute_error(y_test, y_pred_orig),
     "MAPE": np.mean(np.abs((y_test - y_pred_orig)  / y_test)) * 100,
     "R2": r2_score(y_test, y_pred_orig),  "alpha": modelo.alpha_},
    {"modelo": "Ridge_clean", "MAE": mean_absolute_error(y_test, y_pred_clean),
     "MAPE": np.mean(np.abs((y_test - y_pred_clean) / y_test)) * 100,
     "R2": r2_score(y_test, y_pred_clean), "alpha": modelo_clean.alpha_},
]).to_csv(os.path.join(ruta, "metricas_ridge.csv"), index=False)

print("Resultados guardados.\n")

# ------------------------------------------------------------------
# 19. COEFICIENTES DEL MODELO LIMPIO
# ------------------------------------------------------------------
coefs = pd.Series(
    np.abs(modelo_clean.coef_), index=X_train.columns
).sort_values(ascending=False)

print("=== TOP 10 VARIABLES MÁS INFLUYENTES (modelo limpio) ===")
print(coefs.head(10).to_string(), "\n")

# ------------------------------------------------------------------
# 20. RESUMEN ESTADÍSTICO (aproximado — Ridge sesga SE)
# ------------------------------------------------------------------
n = X_train_clean.shape[0]
p = X_train_clean.shape[1]
resid_clean_tr = y_train_clean - modelo_clean.predict(X_train_clean)
mse_clean_tr   = np.sum(resid_clean_tr ** 2) / (n - p - 1)

XtX_inv  = np.linalg.inv(X_train_clean.T @ X_train_clean)
se_betas = np.sqrt(np.diag(XtX_inv * mse_clean_tr))
t_values = modelo_clean.coef_ / se_betas
p_values = 2 * stats.t.sf(np.abs(t_values), df=n - p - 1)
ci_low   = modelo_clean.coef_ - 1.96 * se_betas
ci_high  = modelo_clean.coef_ + 1.96 * se_betas

def sig_stars(p_):
    if p_ < 0.001: return "***"
    if p_ < 0.01:  return "** "
    if p_ < 0.05:  return "*  "
    if p_ < 0.1:   return ".  "
    return "   "

print("=" * 78)
print("RESUMEN ESTADÍSTICO — Ridge (modelo sin outliers)")
print("=" * 78)
hdr = (f"\n{'Variable':<35} {'Estimate':>10} {'Std. Error':>11} "
       f"{'t value':>9} {'Pr(>|t|)':>10} {'CI 95% Low':>11} {'CI 95% High':>12}  Sig.")
print(hdr)
print("-" * len(hdr))

se_int     = np.sqrt(mse_clean_tr / n)
t_int      = modelo_clean.intercept_ / se_int
p_int      = 2 * stats.t.sf(abs(t_int), df=n - p - 1)
ci_int_low  = modelo_clean.intercept_ - 1.96 * se_int
ci_int_high = modelo_clean.intercept_ + 1.96 * se_int
print(f"{'(Intercept)':<35} {modelo_clean.intercept_:>10.6f} {se_int:>11.6f} "
      f"{t_int:>9.3f} {p_int:>10.4e} {ci_int_low:>11.6f} {ci_int_high:>12.6f}  {sig_stars(p_int)}")

signed_coefs = pd.Series(modelo_clean.coef_, index=X_train.columns)
for feat in coefs.index:
    i = list(X_train.columns).index(feat)
    print(f"{feat:<35} {modelo_clean.coef_[i]:>10.6f} {se_betas[i]:>11.6f} "
          f"{t_values[i]:>9.3f} {p_values[i]:>10.4e} "
          f"{ci_low[i]:>11.6f} {ci_high[i]:>12.6f}  {sig_stars(p_values[i])}")

print("-" * len(hdr))
print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
print("⚠  SE, t y p-valores son aproximados (Ridge penaliza coeficientes).\n")

ss_res   = np.sum(resid_clean_tr ** 2)
ss_tot   = np.sum((y_train_clean - y_train_clean.mean()) ** 2)
r2_tr    = 1 - ss_res / ss_tot
r2_adj   = 1 - (1 - r2_tr) * (n - 1) / (n - p - 1)
f_stat   = (r2_tr / p) / ((1 - r2_tr) / (n - p - 1))
f_pvalue = stats.f.sf(f_stat, dfn=p, dfd=n - p - 1)

print(f"Residual std error : {np.sqrt(mse_clean_tr):.6f}  (df = {n - p - 1})")
print(f"Multiple R²        : {r2_tr:.6f}")
print(f"Adjusted R²        : {r2_adj:.6f}")
print(f"F-statistic        : {f_stat:,.2f}  (df1={p}, df2={n-p-1})")
print(f"p-value (F-test)   : {f_pvalue:.4e}")
print("=" * 78 + "\n")

# ------------------------------------------------------------------
# 21. DIAGNÓSTICO GRÁFICO — modelo limpio
# ------------------------------------------------------------------
# Residuos en espacio log (test)
resid_test   = y_test_log - y_pred_log_clean
fitted_log   = y_pred_log_clean
mse_test_    = np.mean(resid_test ** 2)
std_resid    = resid_test / np.sqrt(mse_test_)
sqrt_abs     = np.sqrt(np.abs(std_resid))

# Leverage en test via SVD
U_t, s_t, Vt_t = np.linalg.svd(X_test_sc, full_matrices=False)
hat_test = np.clip(np.sum(U_t ** 2, axis=1), 1e-8, 1 - 1e-8)

# Cook's Distance en test
n_t         = len(resid_test)
p_t         = X_test_sc.shape[1]
cooks_test  = (std_resid ** 2 / p_t) * (hat_test / (1 - hat_test) ** 2)
cooks_thresh_t = 4 / n_t

# VIF sobre train limpio
vif_values = np.zeros(p_t)
for i in range(p_t):
    y_i  = X_train_clean[:, i]
    X_i  = np.delete(X_train_clean, i, axis=1)
    beta = np.linalg.lstsq(X_i, y_i, rcond=None)[0]
    ss_r = np.sum((y_i - X_i @ beta) ** 2)
    ss_t = np.sum((y_i - y_i.mean()) ** 2)
    r2_i = 1 - ss_r / ss_t if ss_t > 0 else 0
    vif_values[i] = 1 / (1 - r2_i) if r2_i < 1 else np.inf

vif_series = pd.Series(vif_values, index=X_train.columns).sort_values(ascending=False)

# Q-Q
si = subsample(n_t)
osm, osr = stats.probplot(resid_test[si], dist="norm", fit=False)
sw_stat, sw_p = stats.shapiro(resid_test[si[:min(2000, len(si))]])
n_influyentes = int(np.sum(cooks_test > cooks_thresh_t))

# ── Figura ────────────────────────────────────────────────────────
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

# 1. Residuos vs Fitted
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

# 2. Q-Q Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(osm, osr, s=5, alpha=0.3, c=C_TEAL, edgecolors="none")
slope_qq, int_qq, *_ = stats.linregress(osm, osr)
line_x = np.array([osm[0], osm[-1]])
ax2.plot(line_x, slope_qq * line_x + int_qq, color=C_RED, linewidth=1.3,
         linestyle="--", label="Normal ref.")
ax2.text(0.04, 0.94, f"Shapiro-Wilk W={sw_stat:.4f}\np={sw_p:.2e}",
         transform=ax2.transAxes, fontsize=7.5, color=C_TEXT, va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#D3D1C7"))
ax2.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax2, "2. Normal Q-Q", "Theoretical quantiles", "Sample quantiles")

# 3. Scale-Location
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(fitted_log[si], sqrt_abs[si], s=5, alpha=0.3, c=C_BLUE, edgecolors="none")
try:
    lw2 = lowess(sqrt_abs[si], fitted_log[si], frac=0.3)
    ax3.plot(lw2[:, 0], lw2[:, 1], color=C_AMBER, linewidth=1.6, label="LOWESS")
    ax3.legend(fontsize=7.5, framealpha=0.5)
except Exception:
    pass
style_ax(ax3, "3. Scale-Location", "Fitted values (log)", "√|Std. residuals|")

# 4. Distribución de residuos
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(resid_test, bins=60, color=C_PURPLE, edgecolor="none", alpha=0.7, density=True)
xr = np.linspace(resid_test.min(), resid_test.max(), 300)
ax4.plot(xr, stats.norm.pdf(xr, resid_test.mean(), resid_test.std()),
         color=C_RED, linewidth=1.5, label="Normal teórica")
ax4.axvline(0, color=C_MUTED, linewidth=0.8, linestyle=":")
ax4.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax4, "4. Distribución de residuos", "Residual", "Densidad")

# 5. Cook's Distance
ax5 = fig.add_subplot(gs[1, 1])
MAX_BARS = 2000
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

# 6. VIF
ax6 = fig.add_subplot(gs[1, 2])
colors_vif = [C_RED if v > 10 else (C_AMBER if v > 5 else C_TEAL) for v in vif_series.values]
ax6.barh(range(len(vif_series)), vif_series.values, color=colors_vif, alpha=0.85)
ax6.set_yticks(range(len(vif_series)))
ax6.set_yticklabels(vif_series.index, fontsize=7.5)
ax6.axvline(5,  color=C_AMBER, linewidth=1, linestyle="--", alpha=0.8, label="VIF=5")
ax6.axvline(10, color=C_RED,   linewidth=1, linestyle="--", alpha=0.8, label="VIF=10")
ax6.legend(fontsize=7.5, framealpha=0.5)
style_ax(ax6, "6. VIF — Multicolinealidad", "VIF", "")

# 7. Residuos vs Leverage
ax7 = fig.add_subplot(gs[2, :])
point_colors = np.where(cooks_test[si] > cooks_thresh_t, C_RED, C_PURPLE)
ax7.scatter(hat_test[si], std_resid[si], s=7, alpha=0.3,
            c=point_colors, edgecolors="none")
ax7.axhline(0,  color=C_MUTED, linewidth=0.8, linestyle=":")
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

out_path = "ridge_assumptions.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ Diagnóstico guardado en: {out_path}\n")

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