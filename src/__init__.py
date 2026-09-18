"""
Módulo principal del proyecto HAR
Reconocimiento de Actividad Humana mediante Teléfonos Inteligentes
"""

__version__ = "1.0.0"
__author__ = "Proyecto de Machine Learning"

from .datos import cargar_datos, guardar_reporte_limpieza, DatosHAR
from .models import EntrenadorModelos
from .evaluation import EvaluadorModelos

__all__ = [
    "cargar_datos",
    "guardar_reporte_limpieza",
    "DatosHAR",
    "EntrenadorModelos",
    "EvaluadorModelos",
]
