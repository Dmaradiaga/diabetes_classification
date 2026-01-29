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
        
        # Classification report como texto
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Datos verdaderos')
        plt.xlabel('Datos predichos')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        os.remove("confusion_matrix.png")
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Modelo {name} registrado. Accuracy: {acc:.4f}, F1: {f1:.4f}")

def run_experiment():
    # Inicializar Dagshub
    # Reemplaza con tu repositorio si es diferente
    dagshub.init(repo_owner='Dmaradiaga', repo_name='diabetes_classification', mlflow=True)
    
    # Directorio raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(project_root, "data", "processed-data", "train.parquet")
    test_path = os.path.join(project_root, "data", "processed-data", "test.parquet")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: No se encuentran los datos procesados. Ejecuta dvc repro o los scripts de datos.")
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
