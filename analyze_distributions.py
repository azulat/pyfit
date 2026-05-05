import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
import numpy as np
import sys

def analyze_data(data, distributions=None):
    """
    Analiza una lista de datos y clasifica las distribuciones que mejor se ajustan.
    
    Args:
        data (list or np.array): Los datos a analizar.
        distributions (list, optional): Lista de distribuciones a probar. 
                                        Si es None, usa TODAS las disponibles.
    """
    if distributions is None:
        # get_distributions() devuelve todas las distribuciones (~80)
        distributions = get_distributions()
    
    print(f"Analizando {len(data)} puntos de datos contra {len(distributions)} distribuciones...")
    
    # Inicializar Fitter
    f = Fitter(data, distributions=distributions, timeout=60)
    
    # Ajustar las distribuciones
    f.fit()
    
    # Obtener el resumen (ranking)
    # El resumen por defecto ordena por sumsquare_error
    summary_df = f.summary()
    
    print("\n--- Ranking de Distribuciones (Mejores Ajustes) ---")
    print(summary_df)
    
    # Identificar distribuciones que fallaron
    # f.df_errors contiene los errores de todas las distribuciones intentadas
    all_attempted = distributions
    successful = summary_df.index.tolist()
    failed = [d for d in all_attempted if d not in f.df_errors.index or pd.isna(f.df_errors.loc[d, 'sumsquare_error'])]
    
    # También podemos ver las que están en df_errors pero con error muy alto o NaN
    # Pero usualmente las que no están en el summary o tienen NaN en df_errors son las "fallidas"
    
    print("\n--- Distribuciones que NO cumplieron/fallaron ---")
    if failed:
        for d in failed:
            print(f"- {d}")
    else:
        print("Todas las distribuciones se ajustaron correctamente (dentro de los límites).")
        
    # Mejor distribución
    best_dist = f.get_best(method='sumsquare_error')
    print(f"\nLa mejor distribución encontrada es: {list(best_dist.keys())[0]}")
    print(f"Parámetros: {best_dist}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analiza distribuciones de datos.")
    parser.add_argument("--file", type=str, help="Ruta a un archivo CSV con los datos.")
    parser.add_argument("--column", type=str, help="Nombre de la columna en el CSV (si aplica).")
    parser.add_argument("--common", action="store_true", help="Probar solo las distribuciones COMUNES (~10) en lugar de todas.")
    
    args = parser.parse_args()
    
    # Por defecto ahora usamos todas, a menos que se pida --common
    distributions = get_common_distributions() if args.common else get_distributions()
    
    if args.file:
        try:
            df = pd.read_csv(args.file)
            if args.column:
                data = df[args.column].values
            else:
                data = df.iloc[:, 0].values
            analyze_data(data, distributions=distributions)
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
    else:
        # Ejemplo de uso con datos aleatorios si no se pasan argumentos
        from scipy import stats
        print("No se proporcionó archivo. Generando datos de ejemplo (Gamma)...")
        data_example = stats.gamma.rvs(2, loc=1.5, scale=2, size=1000)
        analyze_data(data_example, distributions=distributions)
