import os
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_split_data():
    """
    Lee el archivo raw, aplica limpieza, divide en train/test y guarda en Parquet.
    """
    # Directorio raíz del proyecto
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.join(os.path.dirname(current_file_path), "..", "..")
    project_root = os.path.abspath(project_root)
    
    raw_path = os.path.join(project_root, "data", "raw", "diabetes_raw.csv")
    
    if not os.path.exists(raw_path):
        print(f"Error: No se encuentra el archivo {raw_path}. Ejecuta extract_data.py primero.")
        return

    # 1. Cargar datos
    print(f"Cargando datos para transformación desde: {raw_path}")
    data_raw = pd.read_csv(raw_path)

    # 2. Proceso de limpieza
    print("Iniciando proceso de limpieza...")
    initial_shape = data_raw.shape
    
    # Eliminar duplicados
    data_raw = data_raw.drop_duplicates()
    
    # Eliminar filas con valores nulos (si las hay)
    data_raw = data_raw.dropna()
    
    print(f"Limpieza completada. Filas originales: {initial_shape[0]}, Filas después de limpieza: {data_raw.shape[0]}")

    # 3. División de datos (Train / Test)
    print("Dividiendo datos en entrenamiento (80%) y prueba (20%)...")
    train_data, test_data = train_test_split(data_raw, test_size=0.2, random_state=42)

    # 4. Definir rutas de guardado Parquet
    processed_dir = os.path.join(project_root, "data", "processed-data")
    
    train_output = os.path.join(processed_dir, "train.parquet")
    test_output = os.path.join(processed_dir, "test.parquet")

    # 5. Guardar en formato Parquet
    print("Guardando archivos en formato Parquet...")
    train_data.to_parquet(train_output, engine='pyarrow', index=False)
    test_data.to_parquet(test_output, engine='pyarrow', index=False)

    print(f"Entrenamiento guardado en: {train_output}")
    print(f"Prueba guardada en: {test_output}")

if __name__ == "__main__":
    clean_and_split_data()
