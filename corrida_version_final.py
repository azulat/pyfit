import pandas as pd
import numpy as np
from scipy.stats import fisk

def ejecutar_corrida_simple():
    # 1. Configuración de parámetros (según StatFit)
    params = {
        'c': 6.22,      # Forma
        'loc': -12,     # Ubicación
        'scale': 34.7   # Escala
    }

    try:
        # 2. Carga de números pseudoaleatorios
        # El archivo debe ser un .csv con una columna de números entre 0 y 1
        df_input = pd.read_csv('numeros.csv', header=None)
        u = df_input.iloc[:, 0].values
        
        print(f"--- Procesando {len(u)} números de numeros.csv ---")

        # 3. Aplicación de la Transformada Inversa (Log-logistic)
        # Esta función ejecuta internamente: x = loc + scale * (u / (1-u))**(1/c)
        demandas_continuas = fisk.ppf(u, **params)
        
        # 4. Ajuste a la realidad del negocio (enteros positivos)
        demandas_finales = np.maximum(0, np.round(demandas_continuas)).astype(int)

        # 5. Guardar resultados
        df_resultado = pd.DataFrame({
            'u_pseudoaleatorio': u,
            'demanda_simulada': demandas_finales
        })
        
        df_resultado.to_csv('resultado_corrida.csv', index=False)
        
        # 6. Breve resumen en pantalla
        print("\n" + "="*30)
        print("CORRIDA COMPLETADA")
        print("="*30)
        print(f"Demanda Promedio: {demandas_finales.mean():.2f}")
        print(f"Demanda Máxima:   {demandas_finales.max()}")
        print(f"Archivo generado: resultado_corrida.csv")
        print("="*30)

    except FileNotFoundError:
        print("Error: No se encontró el archivo 'numeros.csv'. Asegúrate de que esté en la misma carpeta.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    ejecutar_corrida_simple()