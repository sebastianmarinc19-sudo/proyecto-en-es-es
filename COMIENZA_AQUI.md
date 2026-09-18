# 🚀 COMIENZA AQUI - GUÍA RÁPIDA DE INICIO

## ¡BIENVENIDOS AL PROYECTO HAR!

Este es tu guía de inicio rápido. Sigue estos pasos en orden.

---

## ✅ PASO 1: DESCARGAR E INSTALAR REQUISITOS (5 minutos)

### 1.1 Descargar Python (si no lo tienen)
```
Ir a: https://python.org/downloads/
Descargar Python 3.9 o superior
Instalar normalmente
✓ IMPORTANTE: Marcar "Add Python to PATH"
```

### 1.2 Descargar Visual Studio Code
```
Ir a: https://code.visualstudio.com/
Descargar para Windows
Instalar normalmente
```

### 1.3 Instalar extensiones en VS Code
Ver archivo: **SETUP_VSCODE.md** (Sección 2)

---

## ✅ PASO 2: CONFIGURAR AMBIENTE VIRTUAL (5 minutos)

### Abrir VS Code y seguir estos pasos:

1. **File → Open Folder**
2. Selecciona: `proyecto en es es`
3. Presiona: **Ctrl+Ñ** (abre terminal)
4. Ejecuta:
```bash
python -m venv .venv
```

Espera a que termine...

5. Activa el ambiente:
```bash
.venv\Scripts\Activate.ps1
```

✅ Deberías ver `(.venv)` al inicio de la terminal

---

## ✅ PASO 3: INSTALAR LIBRERÍAS (2 minutos)

En la misma terminal:
```bash
pip install -r requirements.txt
```

Espera a que termine... Verás:
```
Successfully installed pandas numpy scikit-learn...
```

---

## ✅ PASO 4: VERIFICAR QUE TODO FUNCIONA (1 minuto)

En la terminal:
```bash
python scripts/00_inicio.py
```

**Resultado esperado:**
```
✓ Python: 3.9.x
✓ Pandas: 2.0.3
✓ NumPy: 1.24.3
✓ Scikit-learn: 1.2.2
✓ Matplotlib: 3.7.1
✓ Seaborn: 0.12.2
✓ XGBoost: 2.0.1
✓ SciPy: 1.11.1

✅ ¡VERIFICACIÓN COMPLETADA!
```

---

## ✅ PASO 5: DESCARGAR EL DATASET (5 minutos)

### 5.1 Descargar dataset del UCI
```
Ir a: https://archive.ics.uci.edu/ml/datasets/
      human+activity+recognition+using+smartphones

Buscar: "Human Activity Recognition Using Smartphones"
Click: "Download: Data Folder"
Descargar el archivo ZIP
```

### 5.2 Extraer en la carpeta correcta
```
1. Descomprimir el ZIP
2. Encontrar carpeta "UCI HAR Dataset" o similar
3. Copiar archivos a: data/raw/
   
Estructura esperada:
data/raw/
├── train/
├── test/
├── activity_labels.txt
├── features.txt
└── README.txt
```

---

## 🎯 AHORA ESTÁS LISTO PARA EMPEZAR

Tu proyecto está completamente configurado. Aquí está la estructura:

```
proyecto en es es/
├── .venv/                 ← Ambiente virtual ✓
├── data/
│   └── raw/              ← Dataset aquí ↑ (paso 5)
├── src/                  ← Código reutilizable ✓
│   ├── datos.py          ← Carga y limpieza de datos
│   ├── models.py         ← Entrenamiento de modelos
│   └── evaluation.py     ← Métricas y gráficos
├── scripts/              ← Scripts principales ✓
│   ├── 00_inicio.py      ✓ (ya ejecutado)
│   ├── 01_exploracion.py ← Próximo paso
│   ├── 02_entrenamiento.py
│   └── 03_evaluacion.py
├── results/              ← Salidas del proyecto
│   ├── modelos/          ← Modelos guardados
│   └── graficos/         ← Gráficos generados
├── README.md             ← Descripción proyecto ✓
├── PLAN_PROYECTO.md      ← Plan completo ✓
├── SETUP_VSCODE.md       ← Configuración VS Code ✓
└── requirements.txt      ← Librerías necesarias ✓
```

---

## 📚 DOCUMENTACIÓN DISPONIBLE

1. **README.md** - Descripción del proyecto
2. **PLAN_PROYECTO.md** - Plan completo en 5 fases
3. **SETUP_VSCODE.md** - Guía de configuración de VS Code
4. **COMIENZA_AQUI.md** - Este archivo (guía rápida)

---

## 🚀 PRÓXIMOS PASOS - ORDEN RECOMENDADO

### Semana 1-2: EXPLORACIÓN
```bash
python scripts/01_exploracion.py
```
**Qué hace**: Analiza el dataset y crea gráficos
**Output**: Gráficos en `results/graficos/`

### Semana 2-3: ENTRENAMIENTO
```bash
python scripts/02_entrenamiento.py
```
**Qué hace**: Prepara datos y entrena modelos
**Output**: Modelos en `results/modelos/`

### Semana 3-4: EVALUACIÓN
```bash
python scripts/03_evaluacion.py
```
**Qué hace**: Evalúa todos los modelos
**Output**: Métricas en `results/metricas.csv`

### Semana 4-5: PRESENTACIÓN
**Para el 50% del curso:**
- Crear presentación en `presentations/`
- Mostrar gráficos y resultados
- Explicar metodología

---

## ❓ PREGUNTAS FRECUENTES

### ¿Dónde está mi ambiente virtual?
```
Carpeta: .venv
Es invisible, aparece cuando ves (.venv) en la terminal
```

### ¿Cómo activo el ambiente cada vez?
```bash
.venv\Scripts\Activate.ps1
```
Hazlo cada vez que abras una terminal nueva.

### ¿Dónde guardo los gráficos?
```
Automáticamente en: results/graficos/
```

### ¿Cómo veo los scripts disponibles?
```
Carpeta: scripts/
Ver archivos 01_, 02_, 03_, 04_...py
```

### ¿Puedo usar Google Colab en lugar de VS Code?
```
Sí, pero VS Code es mejor para control de versiones y código profesional.
```

---

## 📋 CHECKLIST ANTES DE EMPEZAR

- [ ] Python 3.9+ instalado
- [ ] VS Code instalado
- [ ] Extensiones Python/Pylance instaladas
- [ ] Ambiente virtual `.venv` creado
- [ ] `pip install -r requirements.txt` completado
- [ ] `python scripts/00_inicio.py` ejecutado exitosamente
- [ ] Dataset descargado en `data/raw/`
- [ ] Estructuras de carpetas verificadas

---

## 🎓 APRENDERÁS

- Análisis exploratorio de datos (EDA)
- Preprocesamiento y normalización
- Entrenamiento de modelos ML
- Evaluación de modelos
- Visualización de resultados
- Desarrollo profesional en Python

---

## 📞 AYUDA RÁPIDA

| Problema | Solución |
|----------|----------|
| "python no encontrado" | Reinstala Python, marca "Add to PATH" |
| "ModuleNotFoundError" | Activa ambiente: `.venv\Scripts\Activate.ps1` |
| "pip no reconocido" | Python no está en PATH, reinstala |
| ".venv no aparece" | Es una carpeta oculta, verifica con `dir` |
| "Error activando ambiente" | Ejecuta: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## ✨ ¡ESTÁS LISTO!

**Próximo paso:**
1. Asegúrate de que `.venv` está activado
2. Ejecuta: `python scripts/01_exploracion.py`
3. Espera a que se generen los gráficos
4. Abre los gráficos en `results/graficos/`
5. ¡Comienza el viaje del Machine Learning! 🚀

---

**Última actualización**: 2026-09-18
**Versión**: 1.0 - Proyecto Listo para Comenzar
