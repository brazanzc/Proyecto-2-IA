# mi_random_forest.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

class MiRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=11, max_depth=None, max_features='sqrt', random_state=42):
        # 5. Parámetros configurables solicitados en la pauta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        
    def fit(self, X, y):
        self.trees_ = []
        
        # Asegurarnos de trabajar con arrays de numpy para la indexación
        X_np = np.array(X)
        y_np = np.array(y)
        n_samples = X_np.shape[0]
        
        # 6. Utilizar una semilla general para asegurar reproducibilidad
        np.random.seed(self.random_state)
        # Derivar semillas distintas para cada árbol
        tree_seeds = np.random.randint(0, 100000, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            # 1. Muestra bootstrap: muestreo aleatorio CON reemplazo, mismo tamaño que el original
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_np[indices]
            y_boot = y_np[indices]
            
            # 2. y 3. Entrenar árbol independiente considerando un subconjunto aleatorio de características
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features, # Esto asegura la aleatoriedad de características
                random_state=tree_seeds[i]      # Semilla independiente por árbol
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            
        return self
        
    def predict(self, X):
        X_np = np.array(X)
        
        # Recolectar las predicciones de cada uno de los árboles
        # tree_preds tendrá una forma (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X_np) for tree in self.trees_])
        
        # 4. Combinar predicciones mediante votación mayoritaria
        # stats.mode encuentra el valor más frecuente en cada columna (cada muestra)
        majority_votes, _ = stats.mode(tree_preds, axis=0, keepdims=False)
        
        return majority_votes