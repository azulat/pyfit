import pandas as pd
import numpy as np
from scipy.stats import fisk, ks_2samp
import sys

def generador_lineal_congruencial(n, semilla=42):
    """Genera n números pseudoaleatorios en el rango [0, 1]"""
    m = 2**32
    a = 1664525
    c = 1013904223
    x = semilla
    pseudos = []
    for _ in range(n):
        x = (a * x + c) % m
        pseudos.append(x / m)
    return np.array(pseudos)

def ejecutar_simulacion(cantidad_o_archivo):
    # Parámetros Log-logistic (Fisk) obtenidos de StatFit
    # Según imagen: Loglogistic(-12, 6.22, 34.7)
    params = {
        'c': 6.22,      # Shape parameter
        'loc': -12,     # Location (offset)
        'scale': 34.7   # Scale parameter
    }

    # Datos históricos para validación cruzada
    datos_reales = [
        66,56,16,26,20,22,16,45,13,16,30,8,28,30,22,34,26,25,26,27,25,17,27,14,9,11,26,23,26,31,34,19,32,41,16,49,25,31,
        32,17,20,29,34,35,30,37,40,19,53,54,47,15,18,13,18,16,48,54,22,26,20,26,30,22,13,29,36,31,6,23,26,23,13,17,23,12,
        23,25,22,25,18,10,15,15,23,19,32,23,10,16,7,22,3,13,13,29,20,19,32,27,13,15,23,36,39,25,28,7,27,16,26,35,19,41,
        32,7,15,26,46,36,37,14,23,38,23,27,37,42,31,32,12,27,28,45,18,14,32,25,19,20,25,30,13,32,22,39,24,40,31,16,19,15,
        26,17,27,22,22,25,16,37,41,35,18,19,27,11,26,21,18,31,20,22,8,17,18,30,47,11,28,10,36,19,23,20,19,16,19,29,15,27,
        24,14,48,12,13,14,24,38,11,20,19,22,10,13,14,21,17,8,11,18,19,23,18,31,26,23,20,27,18,26,15,12,31,22,26,31,18,18,
        17,9,23,27,25,24,32,36,17,30,23,21,25,9,15,34,39,41
    ]

    try:
        # Lógica de entrada: ¿Es un número o un archivo?
        if cantidad_o_archivo.isdigit():
            n = int(cantidad_o_archivo)
            print(f"--- Generando {n} nuevos números pseudoaleatorios internamente ---")
            u = generador_lineal_congruencial(n)
        else:
            print(f"--- Cargando números desde archivo: {cantidad_o_archivo} ---")
            df = pd.read_csv(cantidad_o_archivo, header=None)
            u = df.iloc[:, 0].values

        # Transformación a Demanda usando Log-logistic (Fisk)
        # .ppf es la función inversa de la CDF (Transformada Inversa)
        simulados = fisk.ppf(u, **params)
        
        # Limpieza de datos para el negocio:
        # 1. Redondear a enteros (no vendes media empanada)
        # 2. Piso en 0 (la demanda no puede ser negativa)
        simulados = np.maximum(0, np.round(simulados)).astype(int)

        # Validación K-S (Compara si la simulación se parece a la realidad)
        _, p_valor = ks_2samp(datos_reales, simulados)

        # Output de resultados
        print("\n" + "="*45)
        print(f"RESULTADOS SIMULACIÓN LOG-LOGISTIC ({len(simulados)} días)")
        print(f"P-Valor: {p_valor:.4f}")
        print(f"Estado: {'ACEPTADA (Confianza 95%)' if p_valor > 0.05 else 'RECHAZADA'}")
        print("="*45)
        print(f"Media Simulada: {np.mean(simulados):.2f}")
        print(f"Media Real:     {np.mean(datos_reales):.2f}")
        print("="*45)

        # Guardar resultados
        pd.DataFrame({'pseudoaleatorio_u': u, 'demanda_simulada': simulados}).to_csv('simulacion_loglogistic.csv', index=False)
        print("\n[OK] Datos guardados en 'simulacion_loglogistic.csv'")

    except Exception as e:
        print(f"Error en la ejecución: {e}")

if __name__ == "__main__":
    # Si no pasas argumento, simula 90 días por defecto
    entrada = sys.argv[1] if len(sys.argv) > 1 else '90'
    ejecutar_simulacion(entrada)