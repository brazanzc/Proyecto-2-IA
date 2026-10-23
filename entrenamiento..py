# entrenamiento.py

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, recall_score, precision_score, accuracy_score, f1_score, confusion_matrix

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from mi_random_forest import MiRandomForest  # Importamos tu clase propia

# Usar el Pipeline de imblearn es OBLIGATORIO para evitar data leakage al balancear
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def ejecutar_experimentos(X_train, y_train):
    # 6. Métrica prioritaria: Maximizar la sensibilidad de la clase 1 (fraudes)
    scorer = make_scorer(recall_score, pos_label=1)
    
    # 2. Validación cruzada estratificada de 5 particiones
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Definir los modelos base y sus grillas de hiperparámetros
    # (Mantenemos las grillas pequeñas para que el PC no tarde horas)
    modelos = {
        'KNN': {
            'modelo': KNeighborsClassifier(),
            'params': {'clasificador__n_neighbors': [3, 5], 'selector__k': [10, 15]}
        },
        'Arbol_Decision': {
            'modelo': DecisionTreeClassifier(random_state=42),
            'params': {'clasificador__max_depth': [5, 10, None], 'selector__k': [10, 15]}
        },
        'SVM': {
            'modelo': SVC(random_state=42),
            'params': {'clasificador__C': [0.1, 1], 'clasificador__kernel': ['linear', 'rbf'], 'selector__k': [10]}
        },
        'Mi_Random_Forest': {
            'modelo': MiRandomForest(random_state=42),
            'params': {'clasificador__n_estimators': [11, 21], 'clasificador__max_depth': [5, 10], 'selector__k': [10, 15]}
        }
    }
    
    modelos_entrenados = {}
    
    # 1. Entrenar los 4 clasificadores con y sin compensación del desbalance (8 experimentos)
    for nombre, config in modelos.items():
        for balancear in [False, True]:
            nombre_exp = f"{nombre}_{'Balanceado' if balancear else 'Original'}"
            print(f"\n--- Iniciando GridSearch para: {nombre_exp} ---")
            
            # Construir los pasos del pipeline
            pasos = [
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=f_classif)) # Selección de características
            ]
            
            if balancear:
                # El SMOTE solo se aplicará al subset de entrenamiento en cada pliegue de la validación cruzada
                pasos.append(('sampler', SMOTE(random_state=42)))
                
            pasos.append(('clasificador', config['modelo']))
            
            # Crear el pipeline con imblearn
            pipeline = ImbPipeline(pasos)
            
            # Ejecutar GridSearchCV
            grid = GridSearchCV(
                pipeline, 
                config['params'], 
                cv=cv_strat, 
                scoring=scorer,
                n_jobs=-1 # Usa todos los núcleos del procesador
            )
            
            grid.fit(X_train, y_train)
            
            # 7. Registrar hiperparámetros y sensibilidad
            print(f"Mejores hiperparámetros: {grid.best_params_}")
            print(f"Sensibilidad Media (CV): {grid.best_score_:.4f} (+/- {grid.cv_results_['std_test_score'][grid.best_index_]:.4f})")
            
            # Guardar el modelo con mejor desempeño
            modelos_entrenados[nombre_exp] = grid.best_estimator_
            
    return modelos_entrenados

def evaluar_en_prueba(modelo, X_test, y_test):
    # Predicción final sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # Cálculo de matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Cálculo de métricas solicitadas
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    exactitud = accuracy_score(y_test, y_pred)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # Tasa de Falsos Negativos
    
    print(f"Sensibilidad (Recall): {sensibilidad:.4f}")
    print(f"Especificidad:         {especificidad:.4f}")
    print(f"Precisión:             {precision:.4f}")
    print(f"F1-score:              {f1:.4f}")
    print(f"Exactitud (Accuracy):  {exactitud:.4f}")
    print(f"Tasa Falsos Negativos: {fnr:.4f}  <-- ¡Esta es la clave del proyecto!")
    print(f"Matriz de Confusión:\n [[TN: {tn} | FP: {fp}]\n  [FN: {fn}  | TP: {tp}]]\n")