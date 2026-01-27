import os
import pandas as pd

# Ruta que contiene los datos originales
DATA_PATH = "/Users/dmaradiaga/Downloads/diabetes.csv"

def setup_data_directories():
    """
    Crea la estructura de carpetas necesaria para el proyecto utilizando os.path.
    """
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.join(os.path.dirname(current_file_path), "..", "..")
    project_root = os.path.abspath(project_root)
    
    data_dir = os.path.join(project_root, "data")
    print(f"Configurando estructura de datos en: {data_dir}")

    # Crear las carpetas requeridas
    subdirs = ["raw","processed-data"]
    for subdir in subdirs:
        path = os.path.join(data_dir, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Directorio verificado/creado: {path}")
    
    return data_dir

def load_raw_data(data_dir):
    """
    Carga los datos desde DATA_PATH y los guarda en data/raw/diabetes_raw.csv.
    Retorna el DataFrame cargado.
    """
    print(f"Iniciando carga desde: {DATA_PATH}")
    
    try:
        raw_data = pd.read_csv(DATA_PATH)

        raw_data.to_csv(data_dir, index=False)
        print(f"Datos raw guardados en: {data_dir}")
        
        return raw_data
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar: {e}")
        return None

if __name__ == "__main__":
    base_data_path = setup_data_directories()
    load_raw_data(base_data_path)
