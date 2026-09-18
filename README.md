# Proyecto: Reconocimiento de Actividad Humana mediante Teléfonos Inteligentes

**Human Activity Recognition (HAR)** - Clasificación de actividades humanas usando sensores de smartphones

## 📋 Información del Proyecto

- **Objetivo**: Entrenar un modelo de Machine Learning para reconocer 6 tipos de actividades humanas
- **Actividades**: Caminando, Subir Escaleras, Bajar Escaleras, Sentado, De Pie, Acostado
- **Datos**: Acelerómetro y Giroscopio de smartphones (561 características por muestra)
- **Dataset**: UCI Human Activity Recognition Dataset (10,299 muestras, 30 voluntarios)
- **Lenguaje**: Python 3.11
- **IDE**: Visual Studio Code

## 🏆 Resultados obtenidos

| Modelo | Exactitud | Precisión | Sensibilidad | Puntaje F1 |
|---|---|---|---|---|
| **Regresión Logística** | **95.45%** | **95.63%** | **95.45%** | **95.44%** |
| SVM | 95.22% | 95.26% | 95.22% | 95.21% |
| Red Neuronal | 94.77% | 95.00% | 94.77% | 94.79% |
| XGBoost | 93.82% | 93.97% | 93.82% | 93.80% |
| Bosque Aleatorio | 92.67% | 92.78% | 92.67% | 92.66% |

Los 5 modelos están entrenados y guardados en `results/modelos/`. Las métricas completas
están en `results/metricas_modelos.csv` y el desglose por actividad en `results/reporte_detallado.txt`.

## 📁 Estructura del Proyecto

```
proyecto en es es/
├── data/
│   └── raw/                     # Dataset descargado del UCI
├── src/                         # Código reutilizable
│   ├── __init__.py
│   ├── datos.py                 # Carga, verificación y limpieza de datos
│   ├── models.py                # Creación y entrenamiento de modelos
│   └── evaluation.py            # Métricas y gráficos de evaluación
├── scripts/                     # Scripts principales (ejecutar en orden)
│   ├── 00_inicio.py             # Verificación de instalación
│   ├── 01_exploracion.py        # Análisis exploratorio (EDA) + limpieza
│   ├── 02_entrenamiento.py      # Entrenar los 5 modelos
│   └── 03_evaluacion.py         # Evaluar y comparar modelos
├── results/                     # Resultados (se generan al ejecutar)
│   ├── modelos/                 # Los 5 modelos entrenados (.pkl)
│   ├── graficos/                # Los 9 gráficos generados
│   ├── normalizador.pkl         # StandardScaler ajustado en el entrenamiento
│   ├── limpieza_datos.txt       # Reporte de verificación de datos
│   ├── metricas_modelos.csv     # Métricas de los 5 modelos
│   ├── reporte_detallado.txt    # Desglose por actividad
│   ├── verificacion_entrenamiento.txt
│   └── verificacion_evaluacion.txt
├── requirements.txt             # Librerías necesarias
├── PLAN_PROYECTO.md             # Plan completo por fases
└── SETUP_VSCODE.md              # Guía de configuración
```

## 🚀 Quick Start

### 1. Crear ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Verificar instalación
```bash
python scripts/00_inicio.py
```

### 4. Ejecutar el pipeline completo
```bash
python scripts/01_exploracion.py
python scripts/02_entrenamiento.py
python scripts/03_evaluacion.py
```

Los scripts calculan sus rutas a partir de su propia ubicación, así que funcionan
sin importar desde qué carpeta se ejecuten.

## 📊 Fases del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Exploración y análisis (EDA) | ✅ Completada |
| 2 | Limpieza y preparación de datos | ✅ Completada |
| 3 | Desarrollo de modelos (5 algoritmos) | ✅ Completada |
| 4 | Evaluación y validación | ✅ Completada |
| 5 | Presentación (50%) | ⏳ En curso |

## ✅ Verificación automática

Cada script comprueba su propio trabajo y se detiene si algo falla, en vez de
producir resultados silenciosamente incorrectos:

- `01_exploracion.py` → verifica valores faltantes, duplicados, etiquetas y rango de valores
- `02_entrenamiento.py` → 26 verificaciones (consistencia de datos, normalización, cada modelo guardado)
- `03_evaluacion.py` → 15 verificaciones (métricas en rango válido, archivos generados)

Los resultados quedan registrados en `results/verificacion_*.txt`.

## 📚 Recursos

- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [Kaggle Datasets](https://www.kaggle.com)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

## 📝 Notas

- El dataset UCI ya viene normalizado al rango [-1, 1] y sin valores faltantes;
  aun así el pipeline lo verifica y limpia en cada ejecución.
- Los nombres de actividad del dataset vienen en inglés (WALKING, SITTING...);
  se traducen a español únicamente para los gráficos.
- Guías incluidas en la carpeta superior: `Clase 1 - Ppt Generalidades...pdf`,
  `Guia aprendepython.pdf`, `Regresión lineal...pdf`

---

**Última actualización**: 2026-09-18
