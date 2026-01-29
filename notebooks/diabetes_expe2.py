import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def train_and_log_model(model, name, X_train, y_train, X_test, y_test, scaler=None):
    mlflow.set_experiment("diabetes_classification_experiment2")
    mlflow.set_tracking_uri("https://dagshub.com/Dmaradiaga/diabetes_classification.mlflow")
    
    with mlflow.start_run(run_name=name):
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Registrar parámetros y métricas
        params = model.get_params()
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Reporte de clasificación como json (aunque sea texto por ahora)
        report = classification_report(y_test, y_pred)
        viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "visualization"))
        os.makedirs(viz_dir, exist_ok=True)
        report_path = os.path.join(viz_dir, "classification_report_experiment2.json")
        matrix_path = os.path.join(viz_dir, "confusion_matrix_experiment2.png")
        
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens') # Diferente color para diferenciar
        plt.title(f'Confusion Matrix - {name} (Normalized)')
        plt.ylabel('Datos verdaderos')
        plt.xlabel('Datos predichos')
        plt.savefig(matrix_path)
        mlflow.log_artifact(matrix_path)
        plt.close()
       
        
        # Signature e Input Example del modelo
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[0:1]
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            model, 
            name, 
            signature=signature, 
            input_example=input_example
        )
        
        # Registrar escalador si existe
        if scaler:
            scaler_path = "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path)
            os.remove(scaler_path)
            
        print(f"Modelo {name} registrado con signature y normalización. Accuracy: {acc:.4f}, F1: {f1:.4f}")

def run_experiment():
    # Inicializar Dagshub
    dagshub.init(repo_owner='Dmaradiaga', repo_name='diabetes_classification', mlflow=True)
    
    # Directorio raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(project_root, "data", "processed-data", "train.parquet")
    test_path = os.path.join(project_root, "data", "processed-data", "test.parquet")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: No se encuentran los datos procesados. Ejecuta dvc repro.")
        return

    # Cargar datos
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    X_train = train_df.drop(columns=['Outcome'], axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop(columns=['Outcome'], axis=1)
    y_test = test_df['Outcome']
    
    # --- NORMALIZACIÓN ---
    print("Normalizando datos con StandardScaler...")
    scaler = StandardScaler()
    # Identificar columnas numéricas (excluyendo el target que ya quitamos)
    cols_to_scale = X_train.columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Experimentos
    models = [
        (LogisticRegression(max_iter=1000, random_state=42), "Regresion_logistica_Normalizada"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Bosques_aleatorios_Normalizada")
    ]
    
    for model, name in models:
        train_and_log_model(model, name, X_train_scaled, y_train, X_test_scaled, y_test, scaler=scaler)

if __name__ == "__main__":
    run_experiment()
