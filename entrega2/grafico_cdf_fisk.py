"""
Gráfico CDF empírica vs CDF teórica (Fisk) para la demanda de Caprese.

Para cada grupo (Lunes a Viernes y Sábados y Domingos) dibuja:
  - la CDF empírica de los datos reales (escalera: sube 1/n por dato),
  - la CDF teórica de la Fisk con los parámetros de Stat::Fit (curva suave),
  - la distancia D del K-S (la máxima separación vertical entre ambas), que es
    justo el estadístico que mide la prueba de bondad de ajuste.

Si las dos curvas van casi pegadas, la Fisk describe bien los datos.

Entorno headless: usa backend 'Agg' y guarda un PNG (no abre ventana).

Uso:
    python3 grafico_cdf_fisk.py [archivo.csv]   # default: capresse.csv del repo
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda a archivo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisk

# Reutilizamos datos/parámetros/clasificación de la verificación (misma carpeta).
from verificacion_fisk import PARAMS, CSV_DEFAULT, clasificar_grupo

# El PNG se guarda junto a este script.
SALIDA = Path(__file__).resolve().parent / "cdf_fisk.png"


def cdf_empirica(data):
    """Devuelve (x, y) de la CDF empírica (escalón)."""
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def distancia_ks(data, params):
    """
    Distancia D del K-S y el punto (x*, ECDF, CDF teórica) donde ocurre.
    D = máx |ECDF(x) - F(x)| evaluada en los datos.
    """
    x = np.sort(data)
    n = len(x)
    teo = fisk.cdf(x, params["c"], loc=params["loc"], scale=params["scale"])
    ecdf_sup = np.arange(1, n + 1) / n   # ECDF justo a la derecha de cada dato
    ecdf_inf = np.arange(0, n) / n       # ECDF justo a la izquierda
    gaps = np.maximum(np.abs(ecdf_sup - teo), np.abs(teo - ecdf_inf))
    i = int(np.argmax(gaps))
    return gaps[i], x[i], ecdf_sup[i], teo[i]


def graficar(archivo=CSV_DEFAULT):
    df = pd.read_csv(archivo)
    col = df.columns[2]  # 'Uds. vendidas'
    df = df.dropna(subset=[df.columns[0], col]).copy()
    df["grupo"] = df[df.columns[0]].apply(clasificar_grupo)

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(13, 5))
    if len(PARAMS) == 1:
        axes = [axes]

    for ax, (grupo, params) in zip(axes, PARAMS.items()):
        data = df.loc[df["grupo"] == grupo, col].astype(float).values
        if len(data) == 0:
            continue

        # CDF empírica (escalera)
        ex, ey = cdf_empirica(data)
        ax.step(ex, ey, where="post", color="#1f77b4", lw=1.8,
                label=f"CDF empírica (n={len(data)})")

        # CDF teórica de la Fisk (curva suave)
        xs = np.linspace(min(data.min(), 0), data.max() * 1.05, 400)
        ys = fisk.cdf(xs, params["c"], loc=params["loc"], scale=params["scale"])
        ax.plot(xs, ys, color="#d62728", lw=2,
                label="CDF teórica Fisk")

        # Distancia D del K-S
        D, xstar, e_y, t_y = distancia_ks(data, params)
        ax.vlines(xstar, min(e_y, t_y), max(e_y, t_y), color="green", lw=2.5)
        ax.annotate(f"D = {D:.3f}", xy=(xstar, (e_y + t_y) / 2),
                    xytext=(8, 0), textcoords="offset points",
                    color="green", va="center", fontweight="bold")

        ax.set_title(f"{grupo}\nFisk(c={params['c']}, loc={params['loc']}, "
                     f"scale={params['scale']})")
        ax.set_xlabel("Unidades vendidas por día")
        ax.set_ylabel("Probabilidad acumulada")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    fig.suptitle("Bondad de ajuste Fisk — CDF empírica vs teórica "
                 "(la línea verde es la distancia D del K-S)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(SALIDA, dpi=120)
    print(f"[OK] Gráfico guardado en '{SALIDA.name}'")


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    try:
        graficar(entrada)
    except Exception as e:
        print(f"Error en la ejecución: {e}")
