# CONFIGURACIÓN DE VISUAL STUDIO CODE PARA EL PROYECTO

## 1️⃣ DESCARGAR VS CODE

1. Ir a: https://code.visualstudio.com/
2. Descargar versión Windows
3. Instalar normalmente

## 2️⃣ INSTALAR EXTENSIONES EN VS CODE

Abre VS Code y presiona **Ctrl+Shift+X** (Extensiones)

Busca e instala estas extensiones:

```
1. Python (por Microsoft)
   - Proporciona: Intellisense, debugging, testing

2. Pylance (por Microsoft)
   - Autocompletado y análisis de código mejorado

3. Python Environment Manager
   - Gestor de ambientes virtuales

4. Jupyter (por Microsoft)
   - Para ejecutar código con visualización si lo necesitas

5. Better Comments
   - Colores para diferentes tipos de comentarios

6. Code Runner (opcional)
   - Ejecutar código rápidamente
```

## 3️⃣ ABRIR EL PROYECTO EN VS CODE

1. Abre VS Code
2. **File → Open Folder**
3. Navega a: `C:\Users\57320\OneDrive\Escritorio\sitemas inteligentes\proyecto en es es`
4. Selecciona la carpeta y abre

## 4️⃣ CREAR AMBIENTE VIRTUAL

### Paso 1: Abre terminal en VS Code
Presiona: **Ctrl+Ñ** (o **Ctrl+`** en algunos teclados)

### Paso 2: Crea el ambiente virtual
```bash
python -m venv .venv
```

### Paso 3: Activa el ambiente virtual

**En PowerShell (recomendado):**
```bash
.venv\Scripts\Activate.ps1
```

**Si sale error de permisos:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego ejecuta de nuevo:
```bash
.venv\Scripts\Activate.ps1
```

**En CMD (alternativa):**
```bash
.venv\Scripts\activate.bat
```

✅ Si ves `(.venv)` al inicio de la terminal → ¡Activado correctamente!

## 5️⃣ INSTALAR DEPENDENCIAS

Con el ambiente activado:
```bash
pip install -r requirements.txt
```

Espera a que termine (puede tomar 1-2 minutos)

✅ Si ve `Successfully installed` → ¡Todo listo!

## 6️⃣ SELECCIONAR INTÉRPRETE PYTHON

1. Presiona **Ctrl+Shift+P**
2. Escribe: `Python: Select Interpreter`
3. Presiona Enter
4. Busca y selecciona la que diga `.venv` o similar:
   ```
   ./venv/Scripts/python.exe
   ```

✅ Ahora VS Code usa el ambiente correcto

## 7️⃣ VERIFICAR QUE TODO FUNCIONA

1. Abre la terminal (Ctrl+Ñ)
2. Verifica que ves `(.venv)` al inicio
3. Ejecuta:
```bash
python scripts/00_inicio.py
```

✅ Si ves "✅ ¡VERIFICACIÓN COMPLETADA!" → ¡Proyecto configurado!

---

## 💡 TIPS PARA USAR VS CODE CON PYTHON

### Atajos útiles:

| Atajo | Función |
|-------|---------|
| Ctrl+Ñ | Abre/cierra terminal |
| Ctrl+Shift+P | Paleta de comandos |
| Ctrl+F5 | Ejecuta script actual |
| F5 | Inicia debugging |
| Ctrl+/ | Comenta/descomenta línea |
| Ctrl+Shift+X | Abre extensiones |
| Ctrl+Space | Autocompletado |
| Alt+Up/Down | Mover línea arriba/abajo |

### Ejecutar scripts:

**Opción 1: Click derecho en editor**
1. Click derecho en el código
2. "Run Python File in Terminal"

**Opción 2: Atajo**
- Presiona **Ctrl+F5**

**Opción 3: Terminal**
```bash
python scripts/01_exploracion.py
```

## 🐛 DEBUGGING

Para debuguear un script:

1. Haz click en el margen izquierdo para poner **breakpoints** (puntos rojos)
2. Presiona **F5**
3. VS Code para la ejecución en el breakpoint
4. Puedes inspeccionar variables en el panel de la izquierda

## 📊 PARA VER GRÁFICOS

Cuando ejecutas un script con `plt.show()`:
- Se abrirá una ventana emergente con el gráfico
- También se guardan automáticamente en `results/graficos/`

## ⚙️ CONFIGURACIÓN RECOMENDADA DE USUARIO

**Opcional**: Personaliza VS Code abriendo **Ctrl+,**

Busca y modifica:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.formatOnSave": true,
    "editor.fontSize": 14,
    "terminal.integrated.fontSize": 13
}
```

## 🚀 ESTRUCTURA FINAL ESPERADA

```
proyecto en es es/
├── .venv/                 ← Ambiente virtual (creado)
├── data/
│   └── raw/              ← Dataset aquí
├── src/
│   ├── __init__.py       ✓
│   ├── datos.py          ✓ Carga y limpieza de datos
│   ├── models.py         ✓ Entrenamiento de modelos
│   └── evaluation.py     ✓ Métricas y gráficos
├── scripts/
│   ├── 00_inicio.py      ✓ Verificar instalación
│   ├── 01_exploracion.py ✓ EDA + limpieza
│   ├── 02_entrenamiento.py ✓ Entrenar 5 modelos
│   └── 03_evaluacion.py  ✓ Evaluar y comparar
├── results/              ← Se genera al ejecutar
│   ├── modelos/          ← Los 5 modelos entrenados
│   ├── graficos/         ← Los 9 gráficos
│   └── normalizador.pkl  ← Escalado del entrenamiento
├── README.md             ✓
├── requirements.txt      ✓
├── PLAN_PROYECTO.md      ✓
└── SETUP_VSCODE.md       ✓ (este archivo)
```

## 🔍 SOLUCIÓN DE PROBLEMAS

### "comando python no encontrado"
→ Reinstala Python desde python.org

### "No se activa el ambiente virtual"
→ Ejecuta primero: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### "Módulo no encontrado"
→ Verifica que el ambiente está activado (`(.venv)` debe verse)
→ Ejecuta: `pip install -r requirements.txt`

### "ModuleNotFoundError en scripts"
→ Asegúrate de ejecutar desde la carpeta raíz del proyecto
→ O usa el atajo Ctrl+F5 en VS Code

### VS Code no encuentra el intérprete
→ Presiona Ctrl+Shift+P
→ Busca "Python: Select Interpreter"
→ Elige la opción que diga `.venv`

## ✅ CHECKLIST DE SETUP

- [ ] VS Code instalado
- [ ] Extensiones Python/Pylance instaladas
- [ ] Ambiente virtual creado (`.venv`)
- [ ] Ambiente virtual activado (`(.venv)` en terminal)
- [ ] `pip install -r requirements.txt` completado
- [ ] Intérprete Python seleccionado en VS Code
- [ ] `python scripts/00_inicio.py` ejecutado correctamente
- [ ] Dataset descargado en `data/raw/`
- [ ] Listo para empezar scripts

---

**Una vez completado este setup, están listos para comenzar con el primer script de exploración.**

Próximo paso: Ejecutar `python scripts/01_exploracion.py`
