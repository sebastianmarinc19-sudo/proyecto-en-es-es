"""
Módulo: Carga y limpieza de datos
Descripción: Único lugar donde se cargan, verifican y limpian los datos del
             dataset UCI HAR. Los tres scripts usan este módulo, así que la
             limpieza se aplica siempre, no solo durante el análisis exploratorio.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Las rutas se derivan de la ubicación de ESTE archivo, no del directorio
# desde el que se ejecute el script. Así los scripts funcionan aunque se
# lancen desde otra carpeta (por ejemplo con el botón "Run" de VS Code).
CARPETA_PROYECTO = Path(__file__).parent.parent
CARPETA_DATOS = CARPETA_PROYECTO / "data" / "raw"
CARPETA_RESULTADOS = CARPETA_PROYECTO / "results"
CARPETA_GRAFICOS = CARPETA_RESULTADOS / "graficos"
CARPETA_MODELOS = CARPETA_RESULTADOS / "modelos"
RUTA_NORMALIZADOR = CARPETA_RESULTADOS / "normalizador.pkl"

# El dataset UCI trae los nombres de actividad en inglés (WALKING, SITTING...).
# Esta traducción se usa para que los gráficos se vean en español.
NOMBRES_ACTIVIDADES_ES = {
    1: 'Caminando',
    2: 'Subir Escaleras',
    3: 'Bajar Escaleras',
    4: 'Sentado',
    5: 'De Pie',
    6: 'Acostado',
}


class DatosHAR:
    """Los datos del dataset ya cargados, verificados y limpios"""

    def __init__(self, entrenamiento, actividades_entrenamiento, voluntarios_entrenamiento,
                 prueba, actividades_prueba, voluntarios_prueba,
                 nombres_caracteristicas, nombres_actividades, reporte_limpieza):
        self.entrenamiento = entrenamiento
        self.actividades_entrenamiento = actividades_entrenamiento
        self.voluntarios_entrenamiento = voluntarios_entrenamiento
        self.prueba = prueba
        self.actividades_prueba = actividades_prueba
        self.voluntarios_prueba = voluntarios_prueba
        self.nombres_caracteristicas = nombres_caracteristicas
        self.nombres_actividades = nombres_actividades
        self.reporte_limpieza = reporte_limpieza

    def lista_actividades_en_espanol(self):
        """Nombres de las actividades en español, ordenados por su número (1-6)"""
        return [NOMBRES_ACTIVIDADES_ES[i] for i in sorted(self.nombres_actividades.keys())]


def cargar_datos():
    """
    Carga el dataset UCI HAR, lo verifica y lo limpia si hace falta.

    Returns:
        DatosHAR: los datos listos para usar, más el reporte de limpieza
    """
    carpeta_entrenamiento = CARPETA_DATOS / "train"
    carpeta_prueba = CARPETA_DATOS / "test"

    # entrenamiento / prueba: 561 características por muestra
    # actividades_*: la actividad real (1-6) de cada muestra
    # voluntarios_*: qué persona generó cada muestra
    entrenamiento = pd.read_csv(carpeta_entrenamiento / "X_train.txt", sep='\s+', header=None)
    actividades_entrenamiento = pd.read_csv(carpeta_entrenamiento / "y_train.txt", header=None)[0]
    voluntarios_entrenamiento = pd.read_csv(carpeta_entrenamiento / "subject_train.txt", header=None)[0]

    prueba = pd.read_csv(carpeta_prueba / "X_test.txt", sep='\s+', header=None)
    actividades_prueba = pd.read_csv(carpeta_prueba / "y_test.txt", header=None)[0]
    voluntarios_prueba = pd.read_csv(carpeta_prueba / "subject_test.txt", header=None)[0]

    nombres_caracteristicas = pd.read_csv(CARPETA_DATOS / "features.txt", sep='\s+', header=None)[1].values
    tabla_actividades = pd.read_csv(CARPETA_DATOS / "activity_labels.txt", sep='\s+', header=None)
    nombres_actividades = dict(zip(tabla_actividades[0], tabla_actividades[1]))

    reporte = {
        'faltantes_train': int(entrenamiento.isnull().sum().sum()),
        'faltantes_test': int(prueba.isnull().sum().sum()),
        'duplicados_train': int(entrenamiento.duplicated().sum()),
        'duplicados_test': int(prueba.duplicated().sum()),
        'etiquetas_train_validas': set(np.unique(actividades_entrenamiento)).issubset(nombres_actividades.keys()),
        'etiquetas_test_validas': set(np.unique(actividades_prueba)).issubset(nombres_actividades.keys()),
        'valor_minimo': float(entrenamiento.min().min()),
        'valor_maximo': float(entrenamiento.max().max()),
        'acciones': [],
    }

    if reporte['faltantes_train'] > 0 or reporte['faltantes_test'] > 0:
        medias = entrenamiento.mean()
        entrenamiento = entrenamiento.fillna(medias)
        prueba = prueba.fillna(medias)
        reporte['acciones'].append("Se imputaron valores faltantes con la media de cada característica (train).")

    if reporte['duplicados_train'] > 0:
        entrenamiento = entrenamiento.drop_duplicates()
        actividades_entrenamiento = actividades_entrenamiento.loc[entrenamiento.index]
        voluntarios_entrenamiento = voluntarios_entrenamiento.loc[entrenamiento.index]
        reporte['acciones'].append(f"Se eliminaron {reporte['duplicados_train']} filas duplicadas de train.")

    if reporte['duplicados_test'] > 0:
        prueba = prueba.drop_duplicates()
        actividades_prueba = actividades_prueba.loc[prueba.index]
        voluntarios_prueba = voluntarios_prueba.loc[prueba.index]
        reporte['acciones'].append(f"Se eliminaron {reporte['duplicados_test']} filas duplicadas de test.")

    reporte['dataset_estaba_limpio'] = len(reporte['acciones']) == 0

    return DatosHAR(
        entrenamiento=entrenamiento,
        actividades_entrenamiento=actividades_entrenamiento.values,
        voluntarios_entrenamiento=voluntarios_entrenamiento.values,
        prueba=prueba,
        actividades_prueba=actividades_prueba.values,
        voluntarios_prueba=voluntarios_prueba.values,
        nombres_caracteristicas=nombres_caracteristicas,
        nombres_actividades=nombres_actividades,
        reporte_limpieza=reporte,
    )


def guardar_reporte_limpieza(reporte, ruta_salida=None):
    """Guarda en un archivo de texto el resultado de la verificación y limpieza"""
    if ruta_salida is None:
        ruta_salida = CARPETA_RESULTADOS / "limpieza_datos.txt"

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, 'w', encoding='utf-8') as archivo:
        archivo.write("REPORTE DE LIMPIEZA Y VERIFICACIÓN DE DATOS\n")
        archivo.write("=" * 60 + "\n\n")
        archivo.write(f"Valores faltantes (train): {reporte['faltantes_train']}\n")
        archivo.write(f"Valores faltantes (test):  {reporte['faltantes_test']}\n")
        archivo.write(f"Filas duplicadas (train):  {reporte['duplicados_train']}\n")
        archivo.write(f"Filas duplicadas (test):   {reporte['duplicados_test']}\n")
        archivo.write(f"Etiquetas válidas (train): {'Si' if reporte['etiquetas_train_validas'] else 'No'}\n")
        archivo.write(f"Etiquetas válidas (test):  {'Si' if reporte['etiquetas_test_validas'] else 'No'}\n")
        archivo.write(f"Rango de valores: [{reporte['valor_minimo']:.4f}, {reporte['valor_maximo']:.4f}]\n\n")

        if reporte['dataset_estaba_limpio']:
            archivo.write("Conclusion: dataset limpio, sin valores faltantes ni duplicados.\n")
            archivo.write("No se requirio imputacion ni eliminacion de filas.\n")
        else:
            archivo.write("Conclusion: se encontraron problemas y fueron corregidos:\n")
            for accion in reporte['acciones']:
                archivo.write(f"  - {accion}\n")

    return ruta_salida
