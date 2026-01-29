import os
import pandas as pd
from sklearn.impute import SimpleImputer

def transform_data():
    """
    Lee archivos parquet de data/raw, aplica limpieza (duplicados, imputación)
    y guarda en data/processed-data.
    """
    # Directorio raíz del proyecto
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.join(os.path.dirname(current_file_path), "..", "..")
    project_root = os.path.abspath(project_root)
    
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed-data")
    
    # Asegurar que el directorio de salida existe
    os.makedirs(processed_dir, exist_ok=True)

    files = ["train.parquet", "test.parquet"]
    
    for filename in files:
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Error: No se encuentra el archivo {input_path}. Ejecuta extract_data.py primero.")
            continue

        print(f"--- Procesando {filename} ---")
        # 1. Cargar datos
        df = pd.read_parquet(input_path)
        initial_shape = df.shape
        
        # 2. Eliminar duplicados
        df = df.drop_duplicates()
        duplicates_removed = initial_shape[0] - df.shape[0]
        if duplicates_removed > 0:
            print(f"Se eliminaron {duplicates_removed} filas duplicadas.")

        # 3. Imputación de datos
        # Usamos SimpleImputer (media para numéricos por defecto)
        # Identificamos columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not df.empty:
            imputer = SimpleImputer(strategy='median') # Usamos mediana por ser más robusta
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            print(f"Imputación completada en columnas: {list(numeric_cols)}")

        # 4. Revisión de formato (asegurar tipos correctos)
        # En este dataset específico de diabetes, la mayoría deben ser float o int
        # Podríamos forzar a float si fuera necesario, pero SimpleImputer suele devolver floats.
        
        print(f"Limpieza completada. Filas originales: {initial_shape[0]}, Filas finales: {df.shape[0]}")

        # 5. Guardar en formato Parquet
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Archivo procesado guardado en: {output_path}\n")

if __name__ == "__main__":
    transform_data()
