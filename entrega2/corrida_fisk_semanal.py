"""
Corrida de simulación (historia ficticia) de la demanda de empanadas de Caprese
con la distribución Log-logística (Fisk), estructurada POR SEMANAS.

Cada semana se compone de 7 días: 5 de Lunes a Viernes + 2 de Sábado y Domingo,
y cada grupo usa su propio juego de parámetros de Stat::Fit (los del informe):
    Lunes a Viernes   -> c=6.16, loc=-13.3, scale=35.9
    Sábados y Domingos-> c=5.48, loc=-4.4,  scale=27.1
(se reutilizan desde verificacion_fisk.PARAMS para no duplicar).

Método: transformada inversa. Para cada día se toma un U(0,1) y se aplica
    x = fisk.ppf(u, c, loc, scale)
que es exactamente la función generadora del informe
    x = loc + scale * (u/(1-u))**(1/c)
Luego se redondea a entero y se aplica piso en 0 (demanda entera >= 0).

Esto SOLO genera la historia ficticia; la verificación de bondad de ajuste
está en verificacion_fisk.py.

Fuente de los números pseudoaleatorios U(0,1) (configurable):
    - un NÚMERO  -> cantidad de semanas a generar con el GLC propio.
    - una RUTA   -> carga los U desde ese CSV (una columna, sin header);
                    se arman tantas semanas completas como permitan (7 U c/u).

Uso:
    python3 corrida_fisk_semanal.py                 # 4 semanas con el GLC
    python3 corrida_fisk_semanal.py 8               # 8 semanas con el GLC
    python3 corrida_fisk_semanal.py --semilla 123 8 # GLC con otra semilla
    python3 corrida_fisk_semanal.py ../numeros.csv  # usa los U del archivo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisk

# Reutilizamos los parámetros validados en la verificación (misma carpeta).
from verificacion_fisk import PARAMS

DIAS_POR_SEMANA = 7

# Plantilla de una semana: (nombre del día, grupo de PARAMS). El orden define
# en qué orden se consumen los U(0,1) dentro de cada semana.
SEMANA = [
    ("Lunes", "Lunes a Viernes"),
    ("Martes", "Lunes a Viernes"),
    ("Miércoles", "Lunes a Viernes"),
    ("Jueves", "Lunes a Viernes"),
    ("Viernes", "Lunes a Viernes"),
    ("Sábado", "Sábados y Domingos"),
    ("Domingo", "Sábados y Domingos"),
]

SALIDA = Path(__file__).resolve().parent / "simulacion_fisk_semanal.csv"

# Parámetros por defecto del Generador Lineal Congruencial (GLC).
GLC_M = 2 ** 32
GLC_A = 1664525
GLC_C = 1013904223
GLC_SEMILLA = 42


def generador_lineal_congruencial(n, semilla=GLC_SEMILLA):
    """Genera n números pseudoaleatorios U(0,1) con el GLC propio."""
    x = semilla
    pseudos = []
    for _ in range(n):
        x = (GLC_A * x + GLC_C) % GLC_M
        pseudos.append(x / GLC_M)
    return np.array(pseudos)


def demanda_desde_u(u, grupo):
    """Transformada inversa Fisk para un U(0,1): entero >= 0."""
    p = PARAMS[grupo]
    x = fisk.ppf(u, p["c"], loc=p["loc"], scale=p["scale"])
    return int(max(0, round(x)))


def simular(u):
    """
    Arma semanas completas consumiendo los U de a 7 (5 L-V + 2 S-D).
    Devuelve un DataFrame con una fila por día.
    """
    n_semanas = len(u) // DIAS_POR_SEMANA
    if n_semanas == 0:
        raise ValueError(
            f"Se necesitan al menos {DIAS_POR_SEMANA} números U para una semana; "
            f"se recibieron {len(u)}."
        )
    sobrantes = len(u) - n_semanas * DIAS_POR_SEMANA
    if sobrantes:
        print(f"[aviso] {sobrantes} número(s) U sobrantes ignorados "
              f"(no completan una semana de {DIAS_POR_SEMANA} días).")

    filas = []
    for s in range(n_semanas):
        for d, (dia, grupo) in enumerate(SEMANA):
            ui = u[s * DIAS_POR_SEMANA + d]
            filas.append({
                "semana": s + 1,
                "dia": dia,
                "grupo": grupo,
                "u": ui,
                "demanda": demanda_desde_u(ui, grupo),
            })
    return pd.DataFrame(filas)


def obtener_u(entrada, semilla):
    """Devuelve el array de U(0,1) según la entrada (nº de semanas o ruta CSV)."""
    if str(entrada).isdigit():
        semanas = int(entrada)
        n = semanas * DIAS_POR_SEMANA
        print(f"--- Generando {semanas} semana(s) = {n} días con el GLC "
              f"(semilla={semilla}) ---")
        return generador_lineal_congruencial(n, semilla)
    print(f"--- Cargando números U desde archivo: {entrada} ---")
    df = pd.read_csv(entrada, header=None)
    return df.iloc[:, 0].values


def reportar(detalle):
    """Imprime el resumen semanal y totales, y guarda el detalle diario en CSV."""
    semanal = detalle.groupby("semana").agg(
        demanda_semanal=("demanda", "sum"),
        lun_vie=("demanda", lambda x: x[detalle.loc[x.index, "grupo"] == "Lunes a Viernes"].sum()),
        sab_dom=("demanda", lambda x: x[detalle.loc[x.index, "grupo"] == "Sábados y Domingos"].sum()),
    )

    print("\n" + "=" * 60)
    print(f"CORRIDA FISK POR SEMANAS  ({len(semanal)} semanas, "
          f"{len(detalle)} días)")
    print("=" * 60)
    print(f"{'Semana':>6} {'L-V':>6} {'S-D':>6} {'Total':>7}")
    for sem, fila in semanal.iterrows():
        print(f"{sem:>6} {int(fila['lun_vie']):>6} {int(fila['sab_dom']):>6} "
              f"{int(fila['demanda_semanal']):>7}")
    print("-" * 60)
    print(f"Demanda semanal  -> media: {semanal['demanda_semanal'].mean():.2f}   "
          f"min: {int(semanal['demanda_semanal'].min())}   "
          f"max: {int(semanal['demanda_semanal'].max())}")
    print(f"Demanda diaria   -> media: {detalle['demanda'].mean():.2f}   "
          f"total: {int(detalle['demanda'].sum())}")
    print("=" * 60)

    detalle.to_csv(SALIDA, index=False)
    print(f"\n[OK] Detalle diario guardado en '{SALIDA.name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Corrida Fisk por semanas (5 L-V + 2 S-D) por transformada inversa."
    )
    parser.add_argument(
        "entrada", nargs="?", default="4",
        help="Número de semanas a generar con el GLC, o ruta a un CSV con los U. Default: 4.",
    )
    parser.add_argument(
        "--semilla", type=int, default=GLC_SEMILLA,
        help=f"Semilla del GLC (solo si se generan números). Default: {GLC_SEMILLA}.",
    )
    args = parser.parse_args()

    try:
        u = obtener_u(args.entrada, args.semilla)
        detalle = simular(u)
        reportar(detalle)
    except Exception as e:
        print(f"Error en la ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
