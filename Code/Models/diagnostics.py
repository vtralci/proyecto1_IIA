# =============================================================
#  MÓDULO DE DIAGNÓSTICO TRANSVERSAL — diagnostics.py
#
#  Uso en cualquier modelo:
#
#    from diagnostics import (
#        mae_por_decil,
#        plots_trees,     # XGBoost y Random Forest
#        plots_mlp,       # MLPRegressor
#    )
#
#  Todas las funciones reciben:
#    y_true  : np.ndarray — precios reales en € (escala original)
#    y_pred  : np.ndarray — precios predichos en € (escala original)
#    nombre  : str        — nombre del modelo para títulos/ficheros
#    out_dir : str        — directorio donde guardar los PNGs
#
#  plots_trees recibe adicionalmente:
#    modelo       : modelo entrenado (con .feature_importances_)
#    feature_names: lista de nombres de columnas
#    group_labels : pd.Series con la categoría de cada fila del test set
#                   (property_group_enc o similar)
#    group_map    : dict {int: str} para decodificar las categorías
# =============================================================
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL: MAE Y MAPE POR DECIL DE PRECIO
# ══════════════════════════════════════════════════════════════

def mae_por_decil(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  n_bins: int = 10) -> pd.DataFrame:
    """
    Calcula MAE y MAPE por decil del precio real.

    Por qué deciles y no percentiles finos:
      - Con ~18k muestras en test, cada decil tiene ~360 filas → estimación estable.
      - Percentiles finos (p1, p2, …) tendrían <40 filas → alta varianza.

    Por qué MAPE además de MAE:
      - MAE en € crece mecánicamente con el precio (D10 siempre supera D1).
      - MAPE normaliza el error por el precio real → comparable entre deciles.

    Returns
    -------
    pd.DataFrame con columnas:
        decil, precio_min, precio_max, n, mae_eur, mape_pct,
        mediana_error, p25_error, p75_error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errores_abs  = np.abs(y_true - y_pred)
    errores_pct  = errores_abs / y_true * 100

    # Crear deciles basados en y_true
    etiquetas = pd.qcut(y_true, q=n_bins, labels=False, duplicates="drop")

    filas = []
    for d in sorted(np.unique(etiquetas)):
        mask = etiquetas == d
        filas.append({
            "decil"      : int(d) + 1,
            "precio_min" : y_true[mask].min(),
            "precio_max" : y_true[mask].max(),
            "n"          : mask.sum(),
            "mae_eur"    : errores_abs[mask].mean(),
            "mape_pct"   : errores_pct[mask].mean(),
            "mediana_error": np.median(y_true[mask] - y_pred[mask]),
            "p25_error"  : np.percentile(y_true[mask] - y_pred[mask], 25),
            "p75_error"  : np.percentile(y_true[mask] - y_pred[mask], 75),
        })

    df_out = pd.DataFrame(filas)
    print("\n=== MAE / MAPE POR DECIL DE PRECIO ===")
    print(df_out.to_string(index=False,
          float_format=lambda x: f"{x:,.0f}" if x > 100 else f"{x:.1f}"))
    return df_out


# ══════════════════════════════════════════════════════════════
#  PLOTS PARA MODELOS DE ÁRBOL (XGBoost, Random Forest)
# ══════════════════════════════════════════════════════════════

def plots_trees(y_true: np.ndarray,
                y_pred: np.ndarray,
                nombre: str,
                out_dir: str,
                modelo=None,
                feature_names: list = None,
                group_labels: np.ndarray = None,
                group_map: dict = None,
                n_bins: int = 10):
    """
    Panel de diagnóstico para modelos de árbol.

    Gráficas producidas (2 filas × 3 columnas = 6 paneles):
      [0,0] Precios reales vs predichos + línea perfecta
      [0,1] Residuos vs predichos (heteroscedasticidad)
      [0,2] Distribución de residuos (sesgo/simetría)
      [1,0] MAE por decil (€ absolutos)
      [1,1] MAPE por decil (% del precio real)
      [1,2] MAE por grupo de propiedad (sesgo por tipo)
            Si modelo!=None → sustituido por Top-20 importancias

    Qué se comprueba y por qué:
      - Heteroscedasticidad: los árboles no asumen varianza constante,
        pero si el error crece mucho con el precio indica que el target
        en escala log no está corrigiendo suficientemente.
      - Simetría de residuos: un sesgo sistemático (nube desplazada)
        indica que el modelo sobre- o infraestima sistemáticamente.
      - MAE/MAPE por decil: el núcleo del diagnóstico pedido.
        MAE en € muestra el error absoluto; MAPE permite comparar
        si el modelo es igual de bueno en viviendas baratas y caras.
      - Error por grupo: detecta si el modelo falla más en un tipo
        concreto (ej: 'lujo' con pocas muestras y precios altos).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuos = y_true - y_pred
    df_decil = mae_por_decil(y_true, y_pred, n_bins)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"{nombre} — Panel de Diagnóstico", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── [0,0] Real vs Predicho ──────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.scatter(y_true, y_pred, alpha=0.2, s=7, color="steelblue", rasterized=True)
    lim = max(y_true.max(), y_pred.max())
    ax00.plot([0, lim], [0, lim], "r--", lw=1.5, label="Predicción perfecta")
    ax00.set_xlabel("Precio real (€)")
    ax00.set_ylabel("Precio predicho (€)")
    ax00.set_title("Real vs Predicho")
    ax00.legend(fontsize=8)
    ax00.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax00.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # ── [0,1] Residuos vs Predicho (heteroscedasticidad) ───────
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.scatter(y_pred, residuos, alpha=0.2, s=7, color="coral", rasterized=True)
    ax01.axhline(0, color="black", lw=1.2, ls="--")
    # Línea de tendencia suavizada (rolling median)
    orden = np.argsort(y_pred)
    ventana = max(len(y_pred) // 30, 10)
    med_rol = pd.Series(residuos[orden]).rolling(ventana, center=True).median()
    ax01.plot(y_pred[orden], med_rol, color="darkred", lw=1.8,
              label=f"Mediana móvil (w={ventana})")
    ax01.set_xlabel("Precio predicho (€)")
    ax01.set_ylabel("Residuo real − predicho (€)")
    ax01.set_title("Residuos vs Predicho\n(heteroscedasticidad)")
    ax01.legend(fontsize=8)
    ax01.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax01.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # ── [0,2] Distribución de residuos ─────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.hist(residuos / 1000, bins=60, color="steelblue", edgecolor="white",
              linewidth=0.3, alpha=0.85)
    ax02.axvline(0, color="red", lw=1.5, ls="--", label="Error = 0")
    ax02.axvline(np.median(residuos) / 1000, color="orange", lw=1.5,
                 ls="-", label=f"Mediana: {np.median(residuos)/1000:+.1f}k€")
    ax02.set_xlabel("Residuo (miles €)")
    ax02.set_ylabel("Frecuencia")
    ax02.set_title("Distribución de Residuos\n(simetría / sesgo)")
    ax02.legend(fontsize=8)

    # ── [1,0] MAE por decil (€) ─────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    etiquetas_x = [
        f"D{row.decil}\n{row.precio_min/1e3:.0f}-{row.precio_max/1e3:.0f}k"
        for row in df_decil.itertuples()
    ]
    bars = ax10.bar(etiquetas_x, df_decil["mae_eur"] / 1000,
                    color="steelblue", edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, df_decil["mae_eur"]):
        ax10.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.3,
                  f"{val/1e3:.1f}k", ha="center", va="bottom", fontsize=7)
    ax10.set_xlabel("Decil de precio real")
    ax10.set_ylabel("MAE (miles €)")
    ax10.set_title("MAE por Decil de Precio\n(error absoluto €)")
    ax10.tick_params(axis="x", labelsize=7)

    # ── [1,1] MAPE por decil (%) ────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    colors = ["#d73027" if v > 20 else "#fee090" if v > 12 else "#91cf60"
              for v in df_decil["mape_pct"]]
    bars2 = ax11.bar(etiquetas_x, df_decil["mape_pct"],
                     color=colors, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars2, df_decil["mape_pct"]):
        ax11.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.2,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax11.axhline(12, color="orange", lw=1.2, ls="--", label="Umbral 12%")
    ax11.axhline(20, color="red", lw=1.2, ls="--", label="Umbral 20%")
    ax11.set_xlabel("Decil de precio real")
    ax11.set_ylabel("MAPE (%)")
    ax11.set_title("MAPE por Decil de Precio\n(error relativo %)")
    ax11.legend(fontsize=7)
    ax11.tick_params(axis="x", labelsize=7)

    # ── [1,2] Importancias o MAE por grupo ──────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    if modelo is not None and feature_names is not None:
        # Top-20 importancias de features
        imp = pd.Series(modelo.feature_importances_,
                        index=feature_names).sort_values(ascending=True).tail(20)
        imp.plot(kind="barh", ax=ax12, color="steelblue")
        ax12.set_title("Top 20 Variables Importantes")
        ax12.set_xlabel("Importancia")
        ax12.tick_params(axis="y", labelsize=7)
    elif group_labels is not None and group_map is not None:
        # MAE por grupo de propiedad
        group_labels = np.asarray(group_labels)
        grupos = sorted(np.unique(group_labels))
        mae_grupos = [np.abs(residuos[group_labels == g]).mean() / 1000
                      for g in grupos]
        nombres_grupos = [group_map.get(g, str(g)) for g in grupos]
        ax12.bar(nombres_grupos, mae_grupos, color="steelblue",
                 edgecolor="white", linewidth=0.4)
        for i, v in enumerate(mae_grupos):
            ax12.text(i, v + 0.3, f"{v:.1f}k", ha="center", fontsize=9)
        ax12.set_ylabel("MAE (miles €)")
        ax12.set_title("MAE por Tipo de Propiedad\n(sesgo por categoría)")
    else:
        # Fallback: error por grupo sin mapa
        ax12.text(0.5, 0.5, "Sin datos de grupo\ndisponibles",
                  ha="center", va="center", transform=ax12.transAxes)

    nombre_fichero = nombre.lower().replace(" ", "_")
    ruta_out = os.path.join(out_dir, f"diagnostico_{nombre_fichero}.png")
    plt.savefig(ruta_out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Diagnóstico guardado: {ruta_out}")
    return df_decil


# ══════════════════════════════════════════════════════════════
#  PLOTS PARA MLP
# ══════════════════════════════════════════════════════════════

def plots_mlp(y_true: np.ndarray,
              y_pred: np.ndarray,
              nombre: str,
              out_dir: str,
              modelo=None,
              group_labels: np.ndarray = None,
              group_map: dict = None,
              n_bins: int = 10):
    """
    Panel de diagnóstico para MLPRegressor.

    Gráficas (2 filas × 3 columnas = 6 paneles):
      [0,0] Real vs Predicho
      [0,1] Residuos vs Predicho
      [0,2] QQ-plot de residuos (normality check)
      [1,0] MAE por decil (€)
      [1,1] MAPE por decil (%)
      [1,2] Curva de aprendizaje (loss curve)

    Por qué el QQ-plot para MLP y no para árboles:
      Los árboles no hacen ninguna suposición sobre la distribución
      de los errores → el QQ-plot no tiene valor diagnóstico.
      El MLP minimiza MSE, lo que es óptimo bajo el supuesto implícito
      de errores normales. Si los residuos se desvían severamente de
      la normalidad (colas pesadas, asimetría fuerte), indica que:
        (a) quedan outliers sin filtrar que distorsionan el gradiente, o
        (b) la función de pérdida debería cambiarse (MAE loss en lugar de MSE).

    Por qué la curva de aprendizaje para MLP y no para árboles:
      El MLP entrena iterativamente → la curva muestra si hay
      underfitting (loss alta y plana desde el principio),
      overfitting (train loss baja, val loss sube), o convergencia
      correcta (ambas bajan y se estabilizan juntas).
      Los árboles de RF y XGBoost con early_stopping tienen su propio
      mecanismo de control; su curva es menos informativa aquí.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuos = y_true - y_pred
    df_decil = mae_por_decil(y_true, y_pred, n_bins)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"{nombre} — Panel de Diagnóstico", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── [0,0] Real vs Predicho ──────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.scatter(y_true, y_pred, alpha=0.2, s=7, color="steelblue", rasterized=True)
    lim = max(y_true.max(), y_pred.max())
    ax00.plot([0, lim], [0, lim], "r--", lw=1.5, label="Predicción perfecta")
    ax00.set_xlabel("Precio real (€)")
    ax00.set_ylabel("Precio predicho (€)")
    ax00.set_title("Real vs Predicho")
    ax00.legend(fontsize=8)
    ax00.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax00.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # ── [0,1] Residuos vs Predicho ──────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.scatter(y_pred, residuos, alpha=0.2, s=7, color="coral", rasterized=True)
    ax01.axhline(0, color="black", lw=1.2, ls="--")
    orden = np.argsort(y_pred)
    ventana = max(len(y_pred) // 30, 10)
    med_rol = pd.Series(residuos[orden]).rolling(ventana, center=True).median()
    ax01.plot(y_pred[orden], med_rol, color="darkred", lw=1.8,
              label=f"Mediana móvil (w={ventana})")
    ax01.set_xlabel("Precio predicho (€)")
    ax01.set_ylabel("Residuo real − predicho (€)")
    ax01.set_title("Residuos vs Predicho\n(heteroscedasticidad)")
    ax01.legend(fontsize=8)
    ax01.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax01.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # ── [0,2] QQ-plot de residuos (solo MLP) ────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    residuos_norm = (residuos - residuos.mean()) / residuos.std()
    quantiles_teo, quantiles_obs = stats.probplot(residuos_norm, dist="norm")[0]
    ax02.scatter(quantiles_teo, quantiles_obs, alpha=0.3, s=6,
                 color="steelblue", rasterized=True)
    lim_qq = max(abs(quantiles_teo.min()), abs(quantiles_teo.max()))
    ax02.plot([-lim_qq, lim_qq], [-lim_qq, lim_qq], "r--", lw=1.5,
              label="Normal teórica")
    # Test de normalidad (Kolmogorov-Smirnov sobre muestra aleatoria)
    muestra = np.random.choice(residuos_norm, size=min(2000, len(residuos_norm)),
                               replace=False)
    ks_stat, ks_p = stats.kstest(muestra, "norm")
    ax02.set_xlabel("Cuantiles teóricos (Normal)")
    ax02.set_ylabel("Cuantiles observados (residuos)")
    ax02.set_title(f"QQ-Plot de Residuos\nKS stat={ks_stat:.3f}, p={ks_p:.3f}")
    ax02.legend(fontsize=8)
    nota = ("Colas pesadas → outliers\nno filtrados o pérdida MAE"
            if ks_p < 0.05 else "Residuos ≈ normales ✓")
    ax02.text(0.03, 0.97, nota, transform=ax02.transAxes,
              fontsize=7, va="top", color="darkgreen" if ks_p >= 0.05 else "darkred")

    # ── [1,0] MAE por decil (€) ─────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    etiquetas_x = [
        f"D{row.decil}\n{row.precio_min/1e3:.0f}-{row.precio_max/1e3:.0f}k"
        for row in df_decil.itertuples()
    ]
    bars = ax10.bar(etiquetas_x, df_decil["mae_eur"] / 1000,
                    color="steelblue", edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, df_decil["mae_eur"]):
        ax10.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.3,
                  f"{val/1e3:.1f}k", ha="center", va="bottom", fontsize=7)
    ax10.set_xlabel("Decil de precio real")
    ax10.set_ylabel("MAE (miles €)")
    ax10.set_title("MAE por Decil de Precio\n(error absoluto €)")
    ax10.tick_params(axis="x", labelsize=7)

    # ── [1,1] MAPE por decil (%) ────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    colors = ["#d73027" if v > 20 else "#fee090" if v > 12 else "#91cf60"
              for v in df_decil["mape_pct"]]
    bars2 = ax11.bar(etiquetas_x, df_decil["mape_pct"],
                     color=colors, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars2, df_decil["mape_pct"]):
        ax11.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.2,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax11.axhline(12, color="orange", lw=1.2, ls="--", label="Umbral 12%")
    ax11.axhline(20, color="red", lw=1.2, ls="--", label="Umbral 20%")
    ax11.set_xlabel("Decil de precio real")
    ax11.set_ylabel("MAPE (%)")
    ax11.set_title("MAPE por Decil de Precio\n(error relativo %)")
    ax11.legend(fontsize=7)
    ax11.tick_params(axis="x", labelsize=7)

    # ── [1,2] Curva de aprendizaje (solo MLP) ───────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    if modelo is not None and hasattr(modelo, "loss_curve_"):
        ax12.plot(modelo.loss_curve_, label="Train loss", color="steelblue", lw=1.5)
        if modelo.validation_scores_ is not None:
            val_loss = [1 - s for s in modelo.validation_scores_]
            ax12.plot(val_loss, label="Validation loss",
                      color="coral", lw=1.5, ls="--")
        ax12.set_xlabel("Época")
        ax12.set_ylabel("Pérdida")
        ax12.set_title("Curva de Aprendizaje\n(convergencia del MLP)")
        ax12.legend(fontsize=8)
        # Diagnóstico automático
        if len(modelo.loss_curve_) < 20:
            diag = "⚠ Convergencia muy rápida\n→ posible underfitting"
        elif modelo.loss_curve_[-1] > 0.5 * modelo.loss_curve_[0]:
            diag = "⚠ Loss final alta\n→ posible underfitting"
        else:
            diag = "✓ Convergencia normal"
        ax12.text(0.97, 0.97, diag, transform=ax12.transAxes,
                  ha="right", va="top", fontsize=7,
                  color="darkred" if "⚠" in diag else "darkgreen")
    elif group_labels is not None and group_map is not None:
        group_labels = np.asarray(group_labels)
        grupos = sorted(np.unique(group_labels))
        mae_grupos = [np.abs(residuos[group_labels == g]).mean() / 1000
                      for g in grupos]
        nombres_grupos = [group_map.get(g, str(g)) for g in grupos]
        ax12.bar(nombres_grupos, mae_grupos, color="steelblue",
                 edgecolor="white", linewidth=0.4)
        for i, v in enumerate(mae_grupos):
            ax12.text(i, v + 0.3, f"{v:.1f}k", ha="center", fontsize=9)
        ax12.set_ylabel("MAE (miles €)")
        ax12.set_title("MAE por Tipo de Propiedad")
    else:
        ax12.text(0.5, 0.5, "Pasa `modelo` con loss_curve_\npara ver convergencia",
                  ha="center", va="center", transform=ax12.transAxes, fontsize=9)

    nombre_fichero = nombre.lower().replace(" ", "_")
    ruta_out = os.path.join(out_dir, f"diagnostico_{nombre_fichero}.png")
    plt.savefig(ruta_out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Diagnóstico guardado: {ruta_out}")
    return df_decil
