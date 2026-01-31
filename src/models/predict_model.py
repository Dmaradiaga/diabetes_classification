import mlflow
import pandas as pd
import yaml
import os

def predict(data):
    """
    Realiza predicciones utilizando el modelo registrado en MLflow.
    """
    # Cargar parámetros
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    mlflow_params = params.get("mlflow", {})
    model_name = mlflow_params.get("model_name", "bosque_aleatorio")
    stage = mlflow_params.get("stage", "Staging")

    # Cargar el modelo desde el Model Registry
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Cargando modelo desde: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

    # Realizar predicciones
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Ejemplo de uso con datos de prueba
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    test_path = os.path.join(project_root, "data", "processed-data", "test.parquet")
    
    if os.path.exists(test_path):
        test_df = pd.read_parquet(test_path)
        X_test = test_df.drop(columns=['Outcome'], axis=1)
        y_test = test_df['Outcome']
        
        # Predecir sobre las primeras 5 filas
        sample_data = X_test.head(5)
        preds = predict(sample_data)
        
        if preds is not None:
            print(f"Predicciones para la muestra:\n{preds}")
            print(f"Valores reales:\n{y_test.head(5).values}")
    else:
        print("No se encontraron datos de prueba para realizar una demostración.")
