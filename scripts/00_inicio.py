"""
Proyecto: Reconocimiento de Actividad Humana mediante Teléfonos Inteligentes
Script: 00_inicio.py
Descripción: Verificar que todas las librerías estén instaladas correctamente
"""

import sys
from pathlib import Path

# Fuerza salida UTF-8: algunas consolas Windows usan cp1252/cp850 por
# defecto y no pueden imprimir los caracteres ✓/✗ usados en este script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Agregar la carpeta src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("VERIFICACIÓN DE INSTALACIÓN - Proyecto HAR")
print("="*60 + "\n")

# Verificar Python
print(f"✓ Python: {sys.version.split()[0]}")

# Verificar librerías
try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: NO INSTALADO - {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: NO INSTALADO - {e}")

try:
    from sklearn import __version__ as sklearn_version
    print(f"✓ Scikit-learn: {sklearn_version}")
except ImportError as e:
    print(f"✗ Scikit-learn: NO INSTALADO - {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✓ Matplotlib: {plt.matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: NO INSTALADO - {e}")

try:
    import seaborn as sns
    print(f"✓ Seaborn: {sns.__version__}")
except ImportError as e:
    print(f"✗ Seaborn: NO INSTALADO - {e}")

try:
    import xgboost as xgb
    print(f"✓ XGBoost: {xgb.__version__}")
except ImportError as e:
    print(f"✗ XGBoost: NO INSTALADO - {e}")

try:
    import scipy
    print(f"✓ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: NO INSTALADO - {e}")

print("\n" + "="*60)
print("✅ ¡VERIFICACIÓN COMPLETADA!")
print("="*60)
print("\nSiguiente paso: Descargar el dataset del UCI")
print("Descarga desde: https://archive.ics.uci.edu/ml/datasets/")
print("               human+activity+recognition+using+smartphones\n")
print("Extrae en: data/raw/\n")
