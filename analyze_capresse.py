import pandas as pd
from analyze_distributions import analyze_data

def process_capresse(file_path):
    """
    Carga la tercera columna de capresse.csv y analiza sus distribuciones.
    """
    try:
        # Cargar el CSV
        print(f"Cargando datos de {file_path}...")
        df = pd.read_csv(file_path)
        
        # Obtener la tercera columna (índice 2)
        # Según el head, la tercera columna es 'Uds. vendidas'
        if len(df.columns) < 3:
            print("Error: El archivo no tiene al menos 3 columnas.")
            return
            
        col_name = df.columns[2]
        print(f"Analizando la columna: '{col_name}'")
        
        # Convertir a lista/array eliminando valores nulos
        data = df[col_name].dropna().values
        
        # Llamar a la función del script anterior
        analyze_data(data)
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    process_capresse("capresse.csv")
