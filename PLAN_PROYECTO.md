# PLAN DE EJECUCIÓN: Reconocimiento de Actividad Humana

## 🎯 OBJETIVO PRINCIPAL
Desarrollar un modelo de Machine Learning que pueda reconocer y clasificar 6 tipos de actividades humanas usando datos de sensores de smartphones.

---

## 📋 REQUISITOS ANTES DE EMPEZAR

### ✅ INSTALACIONES TÉCNICAS
- Python 3.8+
- Visual Studio Code
- Pip (gestor de paquetes)

### ✅ LIBRERÍAS (en requirements.txt)
```
pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, scipy
```

### ✅ DATASET
UCI Human Activity Recognition Dataset
- Descargar: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Guardar en: `data/raw/`

---

## 🚀 FASES DEL PROYECTO

### **FASE 1: EXPLORACIÓN (Semana 1-2)**
**Archivo**: `scripts/01_exploracion.py`

Tareas:
- [x] Descargar dataset
- [x] Cargar datos con pandas
- [x] Mostrar estructura (shape, dtypes, head)
- [x] Verificar valores faltantes y duplicados
- [x] Estadísticas descriptivas (describe)
- [x] Gráficos de distribución de actividades
- [x] Correlación entre características
- [x] Histogramas de características

**Outputs**:
- 5 gráficos exploratorios en `results/graficos/`
- `results/limpieza_datos.txt` con el resultado de la verificación

---

### **FASE 2: PREPARACIÓN DE DATOS (Semana 2-3)**
**Archivos**: `src/datos.py` (limpieza) y `scripts/02_entrenamiento.py` (normalización)

Tareas:
- [x] Manejar valores faltantes y duplicados
- [x] Usar el split oficial del dataset (train 7,352 / test 2,947)
- [x] Normalizar/Estandarizar características (`StandardScaler`)
- [x] Verificar balance de clases
- [ ] Feature selection (pendiente, opcional)

**Outputs**:
- Datos limpios en memoria vía `src/datos.py` (los 3 scripts usan la misma función)
- `results/normalizador.pkl` para reutilizar el mismo escalado en la evaluación

> **Nota sobre el split**: el dataset UCI ya viene dividido en train/test por
> voluntario (personas distintas en cada conjunto), así que se respeta esa
> división en lugar de hacer un split aleatorio 70/30.

---

### **FASE 3: DESARROLLO DE MODELOS (Semana 3-4)**
**Archivo**: `scripts/02_entrenamiento.py`

Modelos probados (resultado real en test):
1. **Regresión Logística** - 95.45% ← mejor modelo
2. **SVM** - 95.22%
3. **Red Neuronal** - 94.77%
4. **XGBoost** - 93.82%
5. **Bosque Aleatorio** - 92.67%

Tareas por modelo:
- [x] Crear instancia
- [x] Entrenar con datos de train
- [x] Hacer predicciones en test
- [x] Guardar modelo entrenado

**Outputs**:
- Modelos entrenados en `results/modelos/` (.pkl)
- Normalizador guardado en `results/normalizador.pkl`
- Resumen en `results/entrenamiento_summary.txt`

---

### **FASE 4: EVALUACIÓN Y VALIDACIÓN (Semana 4-5)**
**Archivo**: `scripts/03_evaluacion.py`

Métricas por modelo:
- ✓ Accuracy (Exactitud)
- ✓ Precision (Falsos positivos)
- ✓ Recall (Falsos negativos)
- ✓ F1-Score (Balance)
- ✓ Matriz de Confusión
- ✓ ROC-AUC (si aplica)

Visualizaciones:
- [x] Matrices de confusión de los 3 mejores modelos
- [x] Gráfico comparativo de métricas
- [x] Ranking de modelos por Puntaje F1
- [x] Radar del mejor modelo
- [ ] Análisis de errores (pendiente, opcional)

**Outputs**:
- `results/metricas_modelos.csv` - Tabla comparativa
- 9 gráficos en `results/graficos/`
- `results/reporte_detallado.txt` - Desglose por actividad

---

### **FASE 5: PRESENTACIÓN (Semana 5)**
**Ruta**: `presentations/`

Para el 50% del proyecto:
- [ ] Documentar metodología
- [ ] Mostrar top 3 mejores modelos
- [ ] Explicar resultados
- [ ] Incluir gráficos principales
- [ ] Conclusiones y recomendaciones

Contenido mínimo para 50%:
✓ Descripción del dataset (5-10%)
✓ EDA y análisis exploratorio (10-15%)
✓ Preprocesamiento explicado (10-15%)
✓ 3-4 modelos entrenados (15-20%)
✓ Comparación de resultados (10-15%)

---

## 📊 ESTRUCTURA DE SCRIPTS

### `scripts/00_inicio.py`
Verifica que todo esté instalado correctamente
```bash
python scripts/00_inicio.py
```

### `scripts/01_exploracion.py`
Análisis Exploratorio de Datos (EDA)
```bash
python scripts/01_exploracion.py
```

### `scripts/02_entrenamiento.py`
Preprocesamiento y entrenamiento de modelos
```bash
python scripts/02_entrenamiento.py
```

### `scripts/03_evaluacion.py`
Evaluación y comparación de modelos
```bash
python scripts/03_evaluacion.py
```

---

## 🛠️ PASOS INMEDIATOS

1. **Activar ambiente virtual**
```bash
.venv\Scripts\Activate.ps1
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Verificar instalación**
```bash
python scripts/00_inicio.py
```

4. **Descargar dataset**
- Ir a: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Descargar ZIP
- Extraer en: `data/raw/`

5. **Empezar con exploración**
```bash
python scripts/01_exploracion.py
```

---

## 📈 TIMELINE ESPERADO

| Semana | Fase | % Proyecto |
|--------|------|-----------|
| 1-2 | Exploración | 15-20% |
| 2-3 | Preparación | 25-30% |
| 3-4 | Modelado | 45-50% |
| 4-5 | Evaluación | 65-70% |
| 5 | Presentación | 100% |

**Para 50%**: Completar hasta Evaluación (fin de semana 4)

---

## ✅ CHECKLIST FINAL

### Antes de presentar:
- [x] Mínimo 3 modelos diferentes entrenados → se entrenaron **5**
- [x] Todos los gráficos guardados → **9** en `results/graficos/`
- [x] Reporte de métricas guardado en CSV → `results/metricas_modelos.csv`
- [x] README.md actualizado
- [x] Código comentado
- [x] Datos limpios sin valores faltantes → verificado en cada ejecución
- [x] Modelos normalizados correctamente → `StandardScaler` guardado y reutilizado
- [x] Carpeta `results/` con todos los outputs
- [ ] Presentación preparada ← **único pendiente**

---

## 📚 RECURSOS ÚTILES

- UCI Dataset: https://archive.ics.uci.edu/
- Sklearn Docs: https://scikit-learn.org/
- Pandas Guide: https://pandas.pydata.org/docs/
- Python Tutorial: https://docs.python.org/3/

---

## 🎓 CONCEPTOS CLAVE A ENTENDER

1. **EDA** - Análisis Exploratorio de Datos
2. **Normalización** - Estandarizar características
3. **Train/Test Split** - Dividir datos
4. **Clasificación** - Predecir categoría
5. **Métricas** - Accuracy, Precision, Recall, F1
6. **Matriz de Confusión** - Errores por clase
7. **Validación** - Verificar que modelo generaliza
8. **Overfitting** - Modelo memoriza en lugar de aprender

---

**Última actualización**: 2026-09-18
**Estado**: Proyecto creado, listo para empezar
