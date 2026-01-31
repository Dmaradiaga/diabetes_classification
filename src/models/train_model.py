import os
import pandas as pd
import yaml
import pickle
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_final_model():
    # Cargar parámetros
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    pca_params = params.get("pca", {})
    model_params = params.get("train_model", {})
    
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
    
    # Definir el Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_params.get('n_components', 0.95), random_state=model_params.get('random_state', 42))),
        ('classifier', RandomForestClassifier(random_state=model_params.get('random_state', 42)))
    ])
    
    # Configurar RandomizedSearchCV
    param_dist = model_params.get('param_distributions', {})

    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=model_params.get('n_iter', 20),
        cv=model_params.get('cv', 5),
        verbose=1,
        random_state=model_params.get('random_state', 42),
        n_jobs=-1,
        scoring='accuracy'
    )
    
    print("Iniciando optimización del Pipeline (RandomizedSearchCV)...")
    
    # Entrenar modelo con búsqueda
    random_search.fit(X_train, y_train)
    
    best_pipeline = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Predecir con el mejor modelo
    y_pred = best_pipeline.predict(X_test)
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Guardar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    viz_dir = os.path.join(project_root, "src", "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title('Confusion Matrix - Flujo optimizado')
    plt.ylabel('Datos verdaderos')
    plt.xlabel('Datos predichos')
    matrix_path = os.path.join(viz_dir, "final_confusion_matrix.png")
    plt.savefig(matrix_path)
    plt.close()
    
    # Guardar el mejor modelo con pickle
    model_path = os.path.join(model_dir, "bosque_aleatorio.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Mejor modelo guardado en: {model_path}")
    
    
    print(f"Optimización completada. Accuracy: {acc:.4f}, Best CV Score: {random_search.best_score_:.4f}")

if __name__ == "__main__":
    train_final_model()
