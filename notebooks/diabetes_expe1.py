import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_and_log_model(model, name, X_train, y_train, X_test, y_test):
    mlflow.set_experiment("diabetes_classification_experiment1")
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
        
        # Classification report como json
        report = classification_report(y_test, y_pred)
        viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "visualization"))
        os.makedirs(viz_dir, exist_ok=True)
        report_path = os.path.join(viz_dir, "classification_report_experiment1.json")
        
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Datos verdaderos')
        plt.xlabel('Datos predichos')
        plt.savefig("confusion_matrix_experiment1.png")
        mlflow.log_artifact("confusion_matrix_experiment1.png")
        plt.close()
        os.remove("confusion_matrix_experiment1.png")
        
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
        
        print(f"Modelo {name} registrado con signature. Accuracy: {acc:.4f}, F1: {f1:.4f}")

def run_experiment():
    # Inicializar Dagshub
    # Reemplaza con tu repositorio si es diferente
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
    
    # Experimentos
    models = [
        (LogisticRegression(max_iter=1000, random_state=42), "Regresion_logistica"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Bosques_aleatorios")
    ]
    
    for model, name in models:
        train_and_log_model(model, name, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    run_experiment()
