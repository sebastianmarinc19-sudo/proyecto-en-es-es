"""
Módulo: Modelos de Machine Learning
Descripción: Crear, entrenar, guardar y usar los modelos de clasificación
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from pathlib import Path

# XGBoost solo acepta clases numeradas desde 0 (0, 1, 2...), pero las
# actividades del dataset están numeradas desde 1 (1, 2, 3...). Por eso
# XGBoost necesita un paso extra de codificación que los demás modelos no.
MODELOS_QUE_NECESITAN_CODIFICACION = {'XGBoost'}


class EntrenadorModelos:
    """Crea, entrena y guarda los modelos de Machine Learning"""

    def __init__(self):
        self.modelos = {}
        self.modelos_entrenados = {}
        self.codificadores_etiquetas = {}

    def crear_modelos(self):
        """Crea una instancia de cada modelo a probar"""
        self.modelos = {
            'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
            'Bosque Aleatorio': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False),
            'Red Neuronal': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        print(f"✓ {len(self.modelos)} modelos creados")
        for nombre in self.modelos.keys():
            print(f"  - {nombre}")

    def entrenar_modelo(self, nombre_modelo, datos_entrada, etiquetas):
        """
        Entrena un modelo específico

        Args:
            nombre_modelo (str): Nombre del modelo (debe existir en self.modelos)
            datos_entrada (array): Características de entrenamiento
            etiquetas (array): Actividad correspondiente a cada fila
        """
        if nombre_modelo not in self.modelos:
            print(f"✗ Modelo '{nombre_modelo}' no encontrado")
            return None

        print(f"\nEntrenando {nombre_modelo}...")
        modelo = self.modelos[nombre_modelo]

        if nombre_modelo in MODELOS_QUE_NECESITAN_CODIFICACION:
            codificador = LabelEncoder()
            etiquetas_codificadas = codificador.fit_transform(etiquetas)
            self.codificadores_etiquetas[nombre_modelo] = codificador
            modelo.fit(datos_entrada, etiquetas_codificadas)
        else:
            modelo.fit(datos_entrada, etiquetas)

        self.modelos_entrenados[nombre_modelo] = modelo
        print(f"✓ {nombre_modelo} entrenado exitosamente")

        return modelo

    def entrenar_todos_los_modelos(self, datos_entrada, etiquetas):
        """Entrena, uno por uno, todos los modelos creados con crear_modelos()"""
        if not self.modelos:
            self.crear_modelos()

        print("\n" + "="*50)
        print("ENTRENANDO TODOS LOS MODELOS")
        print("="*50)

        for nombre_modelo in self.modelos.keys():
            self.entrenar_modelo(nombre_modelo, datos_entrada, etiquetas)

        print("\n✓ Todos los modelos entrenados")

    def predecir(self, nombre_modelo, datos):
        """
        Predice la actividad para cada fila de `datos` con un modelo ya entrenado

        Returns:
            array: Actividad predicha (1-6) para cada fila
        """
        if nombre_modelo not in self.modelos_entrenados:
            print(f"✗ Modelo '{nombre_modelo}' no está entrenado")
            return None

        predicciones = self.modelos_entrenados[nombre_modelo].predict(datos)

        # Si el modelo usó un codificador de etiquetas al entrenar (XGBoost),
        # hay que revertir la codificación para devolver actividades 1-6.
        if nombre_modelo in self.codificadores_etiquetas:
            predicciones = self.codificadores_etiquetas[nombre_modelo].inverse_transform(predicciones)

        return predicciones

    def predecir_probabilidad(self, nombre_modelo, datos):
        """Devuelve la probabilidad de cada actividad para cada fila de `datos`"""
        if nombre_modelo not in self.modelos_entrenados:
            print(f"✗ Modelo '{nombre_modelo}' no está entrenado")
            return None

        modelo = self.modelos_entrenados[nombre_modelo]
        if hasattr(modelo, 'predict_proba'):
            return modelo.predict_proba(datos)
        else:
            print(f"⚠ {nombre_modelo} no soporta predecir_probabilidad")
            return None

    def guardar_modelo(self, nombre_modelo, ruta_salida):
        """Guarda un modelo entrenado en disco (formato .pkl)"""
        if nombre_modelo not in self.modelos_entrenados:
            print(f"✗ Modelo '{nombre_modelo}' no está entrenado")
            return

        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        contenido = {
            'modelo': self.modelos_entrenados[nombre_modelo],
            'codificador_etiquetas': self.codificadores_etiquetas.get(nombre_modelo)
        }
        joblib.dump(contenido, ruta_salida)
        print(f"✓ Modelo '{nombre_modelo}' guardado en {ruta_salida}")

    def cargar_modelo(self, nombre_modelo, ruta_entrada):
        """Carga un modelo guardado con guardar_modelo() (y su codificador, si tenía uno)"""
        contenido = joblib.load(ruta_entrada)

        if isinstance(contenido, dict) and 'modelo' in contenido:
            modelo = contenido['modelo']
            if contenido.get('codificador_etiquetas') is not None:
                self.codificadores_etiquetas[nombre_modelo] = contenido['codificador_etiquetas']
        else:
            # Compatibilidad con archivos .pkl guardados en un formato antiguo
            modelo = contenido

        self.modelos_entrenados[nombre_modelo] = modelo
        print(f"✓ Modelo '{nombre_modelo}' cargado desde {ruta_entrada}")
        return modelo

    def guardar_todos_los_modelos(self, carpeta_salida):
        """Guarda todos los modelos entrenados en la carpeta indicada"""
        carpeta_salida = Path(carpeta_salida)
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        for nombre_modelo in self.modelos_entrenados.keys():
            ruta = carpeta_salida / f"{nombre_modelo.replace(' ', '_')}.pkl"
            contenido = {
                'modelo': self.modelos_entrenados[nombre_modelo],
                'codificador_etiquetas': self.codificadores_etiquetas.get(nombre_modelo)
            }
            joblib.dump(contenido, ruta)

        print(f"✓ Todos los modelos guardados en {carpeta_salida}")

    def obtener_modelo(self, nombre_modelo):
        """Devuelve el objeto del modelo ya entrenado"""
        return self.modelos_entrenados.get(nombre_modelo)

    def listar_modelos(self):
        """Imprime todos los modelos y si ya están entrenados o no"""
        print("\nModelos disponibles:")
        for nombre in self.modelos.keys():
            estado = "✓ Entrenado" if nombre in self.modelos_entrenados else "⏳ No entrenado"
            print(f"  - {nombre}: {estado}")
