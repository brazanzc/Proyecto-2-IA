# main.py

import pandas as pd
from procesamiento import cargar_y_preprocesar
from entrenamiento import ejecutar_experimentos, evaluar_en_prueba

def main():
    print("--- INICIANDO PROYECTO DE DETECCIÓN DE FRAUDE ---")
    
    # 1. Cargar datos y hacer la partición 70/30 estratificada
    print("\n1. Cargando datos y dividiendo en Train/Test...")
    # Se asume que el dataset está en la misma carpeta
    X_train, X_test, y_train, y_test = cargar_y_preprocesar('creditcard.csv')
    
    # 2. Entrenar y ajustar hiperparámetros (Validación Cruzada)
    print("\n2. Ejecutando los 8 experimentos (CV de 5 particiones)...")
    # Esto devolverá un diccionario con los 8 mejores modelos ya entrenados
    modelos_entrenados = ejecutar_experimentos(X_train, y_train)
    
    # 3. Evaluación final en el conjunto de prueba
    print("\n3. Evaluando modelos en el conjunto de prueba...")
    for nombre_modelo, modelo in modelos_entrenados.items():
        print(f"\n--- Evaluando: {nombre_modelo} ---")
        evaluar_en_prueba(modelo, X_test, y_test)
        
    print("\n--- PROCESO COMPLETADO ---")

if __name__ == "__main__":
    main()