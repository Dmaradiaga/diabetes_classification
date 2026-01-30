import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV


def train_and_log_pca_model(name, model, param_dist, X_train, y_train, X_test, y_test, n_components):
    mlflow.set_experiment("diabetes_classification_experiment4")
    mlflow.set_tracking_uri("https://dagshub.com/Dmaradiaga/diabetes_classification.mlflow")
    
    with mlflow.start_run(run_name=name):
        # Registrar n_components como un parámetro
        mlflow.log_param("pca_n_componentes", n_components)
        
        # Búsqueda de hiperparámetros
        print(f"Iniciando RandomizedSearchCV para {name}...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # Predecir con el mejor modelo
        y_pred = best_model.predict(X_test)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Registrar parámetros del mejor modelo y métricas
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_score", random_search.best_score_)
        
        # Reporte de clasificación como json
        report = classification_report(y_test, y_pred)
        viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "visualization"))
        os.makedirs(viz_dir, exist_ok=True)
        report_path = os.path.join(viz_dir, "classification_report_experiment4.json")
        matrix_path = os.path.join(viz_dir, "confusion_matrix_experiment4.png")
        
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} (PCA & Tuned)')
        plt.ylabel('Datos verdaderos')
        plt.xlabel('Datos predichos')
        plt.savefig(matrix_path)
        mlflow.log_artifact(matrix_path)
        plt.close()
        
        # Signature e Input Example del modelo
        signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
        input_example = X_train[0:1] 
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            best_model, 
            name, 
            signature=signature, 
            input_example=input_example
        )
        
        print(f"Modelo {name} optimizado con PCA y registrado. Accuracy: {acc:.4f}, PCA Components: {n_components}")

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
    
    # Normalización (Requerido para PCA)
    print("Normalizando datos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    print("Aplicando PCA (95% de varianza explicada)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    n_components = pca.n_components_
    print(f"Número de componentes seleccionados: {n_components}")

    
    # Logística
    lr_params = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l2']
    }
    
    # Random Forest
    rf_params = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Experimentos
    models = [
        ("Regresion_logistica_PCA", LogisticRegression(max_iter=1000, random_state=42), lr_params),
        ("Bosques_aleatorios_PCA", RandomForestClassifier(random_state=42), rf_params)
    ]
    
    for name, model, params in models:
        train_and_log_pca_model(name, model, params, X_train_pca, y_train, X_test_pca, y_test, n_components)

if __name__ == "__main__":
    run_experiment()
