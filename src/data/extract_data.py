import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_params(params_path):
    """
    Carga los parámetros desde params.yaml.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f"Parámetros cargados desde {params_path}")
        return params
    except Exception as e:
        print(f"Error al cargar params.yaml: {e}")
        return None

def setup_data_directories():
    """
    Crea la estructura de carpetas necesaria para el proyecto.
    """
    # Directorio raíz del proyecto
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.join(os.path.dirname(current_file_path), "..", "..")
    project_root = os.path.abspath(project_root)
    
    data_dir = os.path.join(project_root, "data")
    print(f"Configurando estructura de datos en: {data_dir}")

    # Crear las carpetas requeridas
    subdirs = ["raw", "processed-data"]
    for subdir in subdirs:
        path = os.path.join(data_dir, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Directorio verificado/creado: {path}")
    
    return data_dir

def load_raw_data(data_dir):
    """
    Carga los datos desde DATA_PATH y los guarda en data/raw/diabetes_raw.csv.
    """
    # Ruta que contiene los datos originales
    DATA_PATH = r"/Users/dmaradiaga/Downloads/diabetes.csv"
    print(f"Iniciando carga desde: {DATA_PATH}")
    
    try:
        raw_data = pd.read_csv(DATA_PATH)
        print("Primeras 5 filas de los datos:")
        print(raw_data.head())
        
        # Guardar en data/raw/diabetes_raw.csv
        output_file = os.path.join(data_dir, "raw", "diabetes_raw.csv")
        raw_data.to_csv(output_file, index=False)
        print(f"Datos raw guardados en: {output_file}")
        
        return raw_data
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar: {e}")
        return None

def split_data(df, test_size, random_state):
    """
    Divide los datos en entrenamiento y prueba.
    """
    print(f"Dividiendo datos con test_size={test_size} y random_state={random_state}")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

if __name__ == "__main__":
    # 1. Configurar directorios
    base_data_path = setup_data_directories()
    
    # 2. Cargar parámetros
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, "..", "..", "params.yaml")
    params = load_params(params_path)
    
    if params:
        test_size = params['data_extraction']['test_size']
        random_state = params['data_extraction']['random_state']
        
        # 3. Cargar datos
        df = load_raw_data(base_data_path)
        
        if df is not None:
            # 4. Dividir datos
            train_df, test_df = split_data(df, test_size, random_state)
            
            # 5. Guardar datos
            train_path = os.path.join(base_data_path, "raw", "train.parquet")
            test_path = os.path.join(base_data_path, "raw", "test.parquet")
            
            train_df.to_parquet(train_path, engine='pyarrow', index=False)
            test_df.to_parquet(test_path, engine='pyarrow', index=False)
            
            print(f"Entrenamiento guardado en: {train_path}")
            print(f"Prueba guardada en: {test_path}")
