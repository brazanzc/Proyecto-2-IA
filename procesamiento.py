# procesamiento.py

import pandas as pd
from sklearn.model_selection import train_test_split

def cargar_y_preprocesar(ruta_archivo='creditcard.csv', n_muestras=None):
    # 1. Cargar el conjunto de datos
    df = pd.read_csv(ruta_archivo)
    
    # 2. Exploración solicitada en la pauta
    print("--- EXPLORACIÓN INICIAL ---")
    print(f"Registros totales: {df.shape[0]}, Características: {df.shape[1]}")
    print(f"Valores faltantes totales: {df.isnull().sum().sum()}")
    print(f"Filas duplicadas: {df.duplicated().sum()}")
    
    # Limpieza: Eliminar duplicados si existen (justificación: transacciones idénticas al mismo milisegundo suelen ser errores de captura)
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print(f"Registros tras eliminar duplicados: {df.shape[0]}")
    
    # Si el PC es lento, la pauta permite usar un subconjunto (mínimo 10.000)
    if n_muestras and n_muestras < len(df):
        # Tomamos una muestra manteniendo la proporción de clases
        df, _ = train_test_split(df, train_size=n_muestras, stratify=df['Class'], random_state=42)
        print(f"Se extrajo una muestra de {n_muestras} registros.")
        
    # Distribución de clases
    fraudes = df[df['Class'] == 1].shape[0]
    legitimas = df[df['Class'] == 0].shape[0]
    print(f"Transacciones legítimas (0): {legitimas} ({legitimas/len(df)*100:.3f}%)")
    print(f"Transacciones fraudulentas (1): {fraudes} ({fraudes/len(df)*100:.3f}%)")
    print("---------------------------\n")

    # 3. Separar variables predictoras de la variable objetivo
    X = df.drop(columns=['Class'])
    y = df['Class']

    # 4. Dividir en entrenamiento (70%) y prueba (30%) de forma estratificada
    # El parámetro stratify=y garantiza que el 30% de prueba conserve la misma proporción real de fraudes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test