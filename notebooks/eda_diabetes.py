import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def perform_eda():
    """
    Análisis Exploratorio de Datos (EDA) sobre el dataset de diabetes.
    """
    # 1. Configuración de rutas
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    data_path = os.path.join(project_root, "data", "raw", "diabetes_raw.csv")

    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo {data_path}.")
        return

    # 2. Carga de datos
    print("--- Cargando datos ---")
    df = pd.read_csv(data_path)
    print(f"Dataset cargado con éxito. Forma: {df.shape}")

    # 3. Análisis Descriptivo Básico
    print("\n--- Información General ---")
    df.info()

    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe())

    print("\n--- Verificación de Datos Nulos ---")
    print(df.isnull().sum())

    print("\n--- Verificación de Datos Duplicados ---")
    print(f"Cantidad de duplicados: {df.duplicated().sum()}")

    # 4. Frecuencias de la variable objetivo (Outcome)
    print("\n--- Frecuencias de Outcome ---")
    print(df['Outcome'].value_counts())

    # 5. Visualización
    print("\n--- Generando Visualizaciones ---")
    
    # Configurar estilo de gráficas
    sns.set_theme(style="whitegrid")
    
    # Histogramas
    plt.figure(figsize=(12, 10))
    df.hist(bins=20, figsize=(15, 12), color='skyblue', edgecolor='black')
    plt.suptitle("Histogramas de Variables de Diabetes", fontsize=16)
    plt.show()

    # Correlación de variables
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación")
    plt.show()

    # 6. Imputación de Datos (Valores faltantes disfrazados de 0)
    # En este dataset, Glucose, BloodPressure, etc. no deberían tener valor 0.
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\n--- Imputación de Datos ---")
    print("Conteo de ceros en columnas físicas (posibles faltantes):")
    for col in cols_to_impute:
        zeros_count = (df[col] == 0).sum()
        print(f"{col}: {zeros_count} ceros")

    # SimpleImputer de Scikit-Learn (imputando con la mediana para evitar impacto de outliers)
    imputer = SimpleImputer(strategy='median')
    
    # Imputación solo si hay ceros (reemplazando temporalmente 0 por NaN)
    import numpy as np
    df_imputed = df.copy()
    df_imputed[cols_to_impute] = df_imputed[cols_to_impute].replace(0, np.nan)
    df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])

    print("\nImputación completada con la mediana de cada columna.")
    print("Conteo de ceros después de imputación:")
    for col in cols_to_impute:
        print(f"{col}: {(df_imputed[col] == 0).sum()} ceros")

    print("\nEDA finalizado.")

if __name__ == "__main__":
    perform_eda()
