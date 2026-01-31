import mlflow
import mlflow.sklearn
import yaml
import os
import pickle
import dagshub
import pandas as pd

def register_model():
    # Cargar parámetros
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Inicializando dagshub
    dagshub.init(repo_owner='Dmaradiaga', repo_name='diabetes_classification', mlflow=True)
    
    mlflow_params = params.get("mlflow", {})
    model_name = mlflow_params.get("model_name", "bosque_aleatorio_final")
    experiment_name = mlflow_params.get("experiment_name", "Diabetes_clasificacion_Final")
    stage = mlflow_params.get("stage", "Staging")

    # Configurar experimento
    mlflow.set_experiment(experiment_name)
    
    # Buscar el mejor run (basado en accuracy)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.accuracy DESC"],
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    
    if runs.empty:
        print(f"No se encontraron runs para el experimento {experiment_name}.")
        return

    best_run_id = runs.iloc[0].run_id
    best_accuracy = runs.iloc[0]['metrics.accuracy']
    print(f"Mejor Run ID encontrado: {best_run_id} con Accuracy: {best_accuracy:.4f}")

    # Ruta del modelo guardado localmente
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(project_root, "models", "bosque_aleatorio_final.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return
    
    print(f"Cargando modelo desde: {model_path}")

    # Directorio raíz y rutas
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_path = os.path.join(project_root, "data", "processed-data", "train.parquet")
    test_path = os.path.join(project_root, "data", "processed-data", "test.parquet")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: No se encuentran los datos procesados.")
        return

    
     # Cargar datos
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    X_train = train_df.drop(columns=['Outcome'], axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop(columns=['Outcome'], axis=1)
    y_test = test_df['Outcome']
    
    
    # Cargar el modelo desde pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Registrar el modelo directamente desde el objeto cargado
    print(f"Registrando modelo '{model_name}' en MLflow Model Registry...")
    
    # Crear un nuevo run para registrar el modelo
    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        # Loguear el modelo
        # Inferir signature e input example
        signature = mlflow.models.infer_signature(X_train, y_train)
        input_example = X_train.iloc[:5]    

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=input_example,
            signature=signature
        )
        
        # Loguear la métrica del mejor run para referencia
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.set_tag("best_run_id", best_run_id)
        
        print(f"Modelo registrado exitosamente como '{model_name}'")
    
    # Obtener la última versión registrada
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if model_versions:
        latest_version = max([int(mv.version) for mv in model_versions])
        print(f"Versión registrada: {latest_version}")
        
        # Asignar stage
        client.transition_model_version_stage(
            name=model_name,
            version=str(latest_version),
            stage=stage,
            archive_existing_versions=True
        )
        print(f"Versión {latest_version} movida a Stage: {stage}")
    else:
        print("No se pudo obtener la versión del modelo registrado")

if __name__ == "__main__":
    register_model()
