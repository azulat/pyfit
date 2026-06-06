"""
Verificación de bondad de ajuste de la distribución Log-logística (Fisk)
para la demanda de empanadas de Caprese.

Replica en Python lo que Stat::Fit reportó como "no rechazable": comprueba que
la distribución Fisk se ajusta a los datos reales, separando -como en el informe-
los días de Lunes a Viernes de los Sábados y Domingos, cada grupo con sus
parámetros propios de Stat::Fit.

Pruebas (cuatro, dos pares: versión convencional + versión que corrige el
sesgo "optimista" de haber estimado los parámetros de los mismos datos):
  1) K-S clásico   : datos reales vs CDF teórica de Fisk (una muestra).
  2) K-S bootstrap : p-valor por bootstrap paramétrico (estilo Lilliefors),
                     re-ajustando la Fisk en cada réplica.
  3) Chi² permisivo: frecuencias obs. vs esperadas (Fisk discretizada con
                     corrección de continuidad), gl = celdas - 1.
  4) Chi² estricto : igual, pero gl = celdas - 1 - 3 (resta los 3 parámetros
                     estimados c, loc, scale).

Criterio (igual que el resto del repo): p_valor > 0.05  =>  NO se rechaza
(la distribución es adecuada con 95% de confianza).

Uso:
    python3 verificacion_fisk.py [archivo.csv] [--bootstrap B]
    # --bootstrap 0 desactiva el bootstrap (mucho más rápido).
"""

import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisk, kstest, chi2

# Réplicas por defecto del bootstrap paramétrico del K-S (configurable por CLI).
BOOTSTRAP_B = 300
BOOTSTRAP_SEMILLA = 42

# capresse.csv vive en la raíz del repo (un nivel arriba de entrega2/).
CSV_DEFAULT = Path(__file__).resolve().parent.parent / "capresse.csv"

# Parámetros de Stat::Fit tomados del informe (UTN-Simulación-TPI1).
# fisk en scipy: c = forma, loc = desplazamiento, scale = escala (beta).
PARAMS = {
    "Lunes a Viernes": {"c": 6.16, "loc": -13.3, "scale": 35.9},
    "Sábados y Domingos": {"c": 5.48, "loc": -4.4, "scale": 27.1},
}

DIAS_FIN_DE_SEMANA = {"sabado", "domingo"}


def _normalizar(texto):
    """minúsculas y sin acentos: 'Miércoles' -> 'miercoles'."""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.strip().lower()


def clasificar_grupo(dia):
    """Devuelve el grupo ('Sábados y Domingos' o 'Lunes a Viernes') según el día."""
    return "Sábados y Domingos" if _normalizar(dia) in DIAS_FIN_DE_SEMANA else "Lunes a Viernes"


def prueba_ks(data, params):
    """K-S de una muestra contra la Fisk teórica. Devuelve (D, p_valor)."""
    D, p = kstest(data, "fisk", args=(params["c"], params["loc"], params["scale"]))
    return D, p


def _ks_stat(data, params3):
    """Estadístico D de K-S de los datos contra Fisk(params3=(c, loc, scale))."""
    return kstest(data, "fisk", args=params3)[0]


def prueba_ks_bootstrap(data, B=BOOTSTRAP_B, semilla=BOOTSTRAP_SEMILLA):
    """
    K-S con p-valor por bootstrap paramétrico (estilo Lilliefors).

    ------------------------------------------------------------------------
    ¿QUÉ ES EL BOOTSTRAP? (explicación didáctica)
    ------------------------------------------------------------------------
    El bootstrap es una técnica para estimar un p-valor (o un margen de error)
    SIMULANDO el experimento muchas veces, en vez de usar una fórmula/tabla
    teórica. El nombre viene de "levantarse tirando de los propios cordones":
    fabricamos la respuesta a partir de nuestros propios datos.

    ¿Por qué acá no alcanza el K-S clásico? El K-S clásico saca su p-valor de
    una tabla teórica que SOLO vale si la distribución vino "de afuera". Pero
    nosotros estimamos c, loc y scale de los MISMOS datos, así que esa tabla da
    un p-valor inflado (demasiado fácil de aprobar). Para ese escenario no hay
    tabla simple, así que la fabricamos simulando.

    La pregunta que respondemos: "¿qué tan grande puede ser la distancia D del
    K-S, por puro azar, si los datos de verdad fueran Fisk?". La respondemos
    creando 'universos paralelos':

      1) Medimos lo real: ajustamos Fisk a los datos -> params_hat, y calculamos
         la distancia observada D_obs = K-S(datos, params_hat).
      2) Repetimos B veces (cada vuelta = un universo paralelo):
         - generamos n datos FALSOS que, por construcción, SÍ salen de
           Fisk(params_hat);
         - los tratamos igual que a los reales: les RE-AJUSTAMOS una Fisk
           (params_b) y medimos su distancia D_b. Re-ajustar es clave: imita el
           haber estimado parámetros, y es lo que captura el sesgo "optimista".
         - así, cada D_b es "una distancia que aparece por puro azar cuando el
           modelo es correcto".
      3) p = fracción de esos D_b azarosos que resultó >= D_obs, con la
         corrección +1 para no dar p = 0:  p = (#{D_b >= D_obs} + 1) / (B + 1).

    Lectura: p grande => nuestra distancia real es de lo más común bajo el
    modelo => NO se rechaza. p chico => era rarísima => se rechaza.

    Analogía: para saber si un dado es honesto, en vez de calcular
    probabilidades a mano, agarro un dado que SÉ honesto, lo tiro 300 veces y
    comparo mi dado sospechoso contra esa experiencia.

    (Es lento porque cada universo paralelo requiere re-ajustar una Fisk, y eso
    se hace B veces. Es el precio de tener un p-valor correcto en este caso.)
    ------------------------------------------------------------------------

    Devuelve (D_obs, p_valor, params_hat).
    """
    n = len(data)
    rng = np.random.default_rng(semilla)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params_hat = fisk.fit(data)  # (c, loc, scale) estimados de los datos
        D_obs = _ks_stat(data, params_hat)
        extremos = 0
        for _ in range(B):
            # un "universo paralelo": datos falsos que sí son Fisk(params_hat)...
            muestra = fisk.rvs(*params_hat, size=n, random_state=rng)
            # ...y los tratamos igual que a los reales (re-ajuste + distancia).
            params_b = fisk.fit(muestra)
            if _ks_stat(muestra, params_b) >= D_obs:
                extremos += 1
    p_valor = (extremos + 1) / (B + 1)
    return D_obs, p_valor, params_hat


def _fusionar_bins(obs, esp, min_esp=5.0):
    """Fusiona celdas adyacentes hasta que toda esperada >= min_esp (regla de Cochran)."""
    obs, esp = list(obs), list(esp)
    i = 0
    while i < len(esp):
        if esp[i] < min_esp and len(esp) > 1:
            j = i - 1 if i == len(esp) - 1 else i + 1  # fusiona con vecino
            obs[j] += obs[i]
            esp[j] += esp[i]
            del obs[i]
            del esp[i]
            i = max(0, i - 1)
        else:
            i += 1
    return np.array(obs), np.array(esp)


def prueba_chi2(data, params):
    """
    Chi-cuadrado discretizando la Fisk continua con corrección de continuidad.
    Devuelve (chi2_stat, p_valor, gl, n_celdas).
    """
    n = len(data)
    data = np.round(data).astype(int)
    valores = np.arange(data.min(), data.max() + 1)

    cdf = lambda x: fisk.cdf(x, params["c"], loc=params["loc"], scale=params["scale"])
    # Bordes a la mitad entre enteros; colas abiertas en -inf / +inf para sumar prob = 1.
    bordes = np.concatenate(([-np.inf], valores[:-1] + 0.5, [np.inf]))
    probs = cdf(bordes[1:]) - cdf(bordes[:-1])

    obs = np.array([(data == v).sum() for v in valores], dtype=float)
    esp = n * probs

    obs, esp = _fusionar_bins(obs, esp)
    chi2_stat = float(np.sum((obs - esp) ** 2 / esp))

    # Versión "permisiva": gl = celdas - 1 (toma los parámetros como dados).
    gl = len(obs) - 1
    p = float(chi2.sf(chi2_stat, gl))

    # Versión "estricta": como c, loc y scale se estimaron de los datos, se
    # restan esos 3 grados de libertad (gl = celdas - 1 - 3).
    gl_estricto = max(1, gl - 3)
    p_estricto = float(chi2.sf(chi2_stat, gl_estricto))

    return {
        "chi2": chi2_stat, "celdas": len(obs),
        "gl": gl, "p": p,
        "gl_estricto": gl_estricto, "p_estricto": p_estricto,
    }


def _veredicto(p):
    return "NO se rechaza (adecuada, 95%)" if p > 0.05 else "SE RECHAZA"


def verificar(archivo=CSV_DEFAULT, bootstrap_B=BOOTSTRAP_B):
    df = pd.read_csv(archivo)
    col_demanda = df.columns[2]  # 'Uds. vendidas'
    df = df.dropna(subset=[df.columns[0], col_demanda]).copy()
    df["grupo"] = df[df.columns[0]].apply(clasificar_grupo)

    print("=" * 70)
    print("VERIFICACIÓN DE BONDAD DE AJUSTE — DISTRIBUCIÓN FISK (Log-logística)")
    print(f"Archivo: {archivo}   |   Variable: '{col_demanda}'")
    print("=" * 70)

    for grupo, params in PARAMS.items():
        data = df.loc[df["grupo"] == grupo, col_demanda].astype(float).values
        if len(data) == 0:
            print(f"\n[{grupo}] sin datos, se omite.")
            continue

        D, p_ks = prueba_ks(data, params)
        chi = prueba_chi2(data, params)

        print(f"\n--- {grupo} ---")
        print(f"  n = {len(data)}   media = {data.mean():.2f}   "
              f"min = {int(data.min())}   max = {int(data.max())}")
        print(f"  Fisk(c={params['c']}, loc={params['loc']}, scale={params['scale']})")
        print(f"  K-S clásico:         D = {D:.4f}   p = {p_ks:.4f}   "
              f"=> {_veredicto(p_ks)}")

        if bootstrap_B > 0:
            print(f"  (calculando bootstrap del K-S, B={bootstrap_B}...)", flush=True)
            D_b, p_b, p_hat = prueba_ks_bootstrap(data, bootstrap_B)
            ajuste = f"c={p_hat[0]:.2f}, loc={p_hat[1]:.2f}, scale={p_hat[2]:.2f}"
            print(f"  K-S bootstrap:       D = {D_b:.4f}   p = {p_b:.4f}   "
                  f"=> {_veredicto(p_b)}   [re-ajuste: {ajuste}]")

        print(f"  Chi² permisivo:      X2 = {chi['chi2']:.4f}   gl = {chi['gl']}   "
              f"(celdas = {chi['celdas']})   p = {chi['p']:.4f}   "
              f"=> {_veredicto(chi['p'])}")
        print(f"  Chi² estricto (gl-3):X2 = {chi['chi2']:.4f}   gl = {chi['gl_estricto']}"
              f"                p = {chi['p_estricto']:.4f}   "
              f"=> {_veredicto(chi['p_estricto'])}")

    print("\n" + "=" * 70)
    print("Criterio: p > 0.05 => NO se rechaza la hipótesis de que los datos")
    print("provienen de una distribución Fisk (confianza del 95%).")
    print("El K-S bootstrap y el Chi² estricto corrigen el sesgo optimista de")
    print("haber estimado los parámetros (c, loc, scale) de los mismos datos.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bondad de ajuste de la Fisk a la demanda de Caprese (K-S + Chi²)."
    )
    parser.add_argument("archivo", nargs="?", default=str(CSV_DEFAULT),
                        help="CSV de datos. Default: capresse.csv del repo.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B, metavar="B",
                        help=f"Réplicas del bootstrap del K-S (0 lo desactiva). Default: {BOOTSTRAP_B}.")
    args = parser.parse_args()
    try:
        verificar(args.archivo, bootstrap_B=args.bootstrap)
    except Exception as e:
        print(f"Error en la ejecución: {e}")
