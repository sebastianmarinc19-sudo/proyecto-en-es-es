"""
Módulo: Evaluación de modelos
Descripción: Medir qué tan bien predice cada modelo y compararlos entre sí
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Las métricas se calculan y guardan como Accuracy/Precision/Recall/F1-Score
# (así quedan en el CSV y en el reporte). Esta traducción es solo para que
# los gráficos se muestren en español.
NOMBRES_METRICAS_ES = {
    'Accuracy': 'Exactitud',
    'Precision': 'Precisión',
    'Recall': 'Sensibilidad',
    'F1-Score': 'Puntaje F1',
}


class EvaluadorModelos:
    """Calcula métricas, genera gráficos y compara el rendimiento de los modelos"""

    def __init__(self):
        self.resultados = {}
        self.predicciones = {}

    def evaluar_modelo(self, nombre_modelo, valores_reales, predicciones, probabilidades=None):
        """
        Calcula las métricas de un modelo comparando sus predicciones contra la realidad

        Args:
            nombre_modelo (str): Nombre del modelo evaluado
            valores_reales (array): Actividad verdadera de cada muestra
            predicciones (array): Actividad que predijo el modelo
            probabilidades (array): Probabilidad de cada actividad (opcional)

        Returns:
            dict: Métricas calculadas (Accuracy, Precision, Recall, F1-Score)
        """
        metricas = {
            'Accuracy': accuracy_score(valores_reales, predicciones),
            'Precision': precision_score(valores_reales, predicciones, average='weighted', zero_division=0),
            'Recall': recall_score(valores_reales, predicciones, average='weighted', zero_division=0),
            'F1-Score': f1_score(valores_reales, predicciones, average='weighted', zero_division=0)
        }

        # ROC-AUC solo tiene sentido cuando hay exactamente 2 actividades posibles
        if len(np.unique(valores_reales)) == 2 and probabilidades is not None:
            try:
                metricas['ROC-AUC'] = roc_auc_score(valores_reales, probabilidades[:, 1])
            except Exception:
                metricas['ROC-AUC'] = None

        self.resultados[nombre_modelo] = metricas
        self.predicciones[nombre_modelo] = {'reales': valores_reales, 'predichas': predicciones}

        return metricas

    def _construir_tabla_resultados(self):
        """
        Junta las métricas de todos los modelos en una sola tabla (DataFrame),
        con el nombre del modelo como índice.

        pd.DataFrame(diccionario_de_diccionarios).T puede dejar las columnas
        en un tipo de dato genérico ("object") en vez de numérico, y con ese
        tipo operaciones como nlargest() o idxmax() fallan. Por eso aquí se
        fuerzan a numérico.
        """
        tabla = pd.DataFrame(self.resultados).T
        return tabla.apply(pd.to_numeric, errors='coerce')

    def comparar_modelos(self):
        """Imprime y devuelve una tabla con las métricas de todos los modelos evaluados"""
        if not self.resultados:
            print("✗ No hay modelos evaluados aún")
            return None

        tabla = self._construir_tabla_resultados()
        tabla = tabla.sort_values('F1-Score', ascending=False)

        print("\n" + "="*70)
        print("COMPARACIÓN DE MODELOS")
        print("="*70)
        print(tabla.to_string())
        print("="*70)

        return tabla

    def imprimir_reporte_detallado(self, nombre_modelo, valores_reales, predicciones):
        """Imprime precisión, recall y f1-score actividad por actividad"""
        print(f"\n{'='*60}")
        print(f"REPORTE DETALLADO: {nombre_modelo}")
        print(f"{'='*60}")
        print(f"\nClasificación por actividad:")
        print(classification_report(valores_reales, predicciones))

    def graficar_matriz_confusion(self, nombre_modelo, valores_reales, predicciones, ruta_guardado=None):
        """
        Dibuja la matriz de confusión: qué actividades reales se confundieron
        con cuáles actividades predichas.
        """
        matriz = confusion_matrix(valores_reales, predicciones)

        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.ylabel('Actividad Real')
        plt.xlabel('Actividad Predicha')
        plt.tight_layout()

        if ruta_guardado:
            ruta_guardado = Path(ruta_guardado)
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"✓ Matriz de confusión guardada en {ruta_guardado}")

        plt.show()

    def graficar_comparacion_modelos(self, ruta_guardado=None):
        """Dibuja un gráfico de barras comparando Accuracy, Precision, Recall y F1-Score"""
        if not self.resultados:
            print("✗ No hay modelos para comparar")
            return

        tabla = self._construir_tabla_resultados()

        metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        tabla_metricas = tabla[metricas].rename(columns=NOMBRES_METRICAS_ES)

        plt.figure(figsize=(12, 6))
        tabla_metricas.plot(kind='bar', ax=plt.gca())
        plt.title('Comparación de Modelos - Métricas de Rendimiento')
        plt.ylabel('Puntaje')
        plt.xlabel('Modelo')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)

        if ruta_guardado:
            ruta_guardado = Path(ruta_guardado)
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfica de comparación guardada en {ruta_guardado}")

        plt.show()

    def guardar_resultados_csv(self, ruta_salida):
        """Guarda la tabla de métricas de todos los modelos en un archivo CSV"""
        if not self.resultados:
            print("✗ No hay resultados para guardar")
            return

        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        tabla = self._construir_tabla_resultados()
        tabla.to_csv(ruta_salida, index_label='Modelo')
        print(f"✓ Resultados guardados en {ruta_salida}")

    def obtener_mejor_modelo(self):
        """Devuelve el nombre del modelo con el F1-Score más alto"""
        if not self.resultados:
            return None

        tabla = self._construir_tabla_resultados()
        return tabla['F1-Score'].idxmax()

    def graficar_curva_roc(self, nombre_modelo, valores_reales, probabilidades, ruta_guardado=None):
        """Dibuja la curva ROC (solo válida para clasificación de 2 actividades)"""
        try:
            fpr, tpr, _ = roc_curve(valores_reales, probabilidades[:, 1])
            auc = roc_auc_score(valores_reales, probabilidades[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title(f'Curva ROC - {nombre_modelo}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if ruta_guardado:
                ruta_guardado = Path(ruta_guardado)
                ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
                print(f"✓ Curva ROC guardada en {ruta_guardado}")

            plt.show()
        except Exception as e:
            print(f"⚠ No se pudo graficar ROC: {e}")
