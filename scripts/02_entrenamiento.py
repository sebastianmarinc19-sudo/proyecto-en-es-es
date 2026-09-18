"""
Proyecto: Reconocimiento de Actividad Humana (HAR)
Script: 02_entrenamiento.py
Descripción: Normalizar los datos y entrenar los modelos de Machine Learning
"""

import numpy as np
import sys
import time
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Fuerza salida UTF-8: algunas consolas Windows usan cp1252/cp850 por
# defecto y no pueden imprimir los caracteres ✓/📊/🎯 usados en este script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Agregar la carpeta del proyecto al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datos import cargar_datos, CARPETA_MODELOS, CARPETA_RESULTADOS, RUTA_NORMALIZADOR
from src.models import EntrenadorModelos

print("\n" + "="*70)
print("🚀 ENTRENAMIENTO DE MODELOS - PROYECTO HAR")
print("="*70 + "\n")

# 1. CARGAR DATOS
print("1️⃣  Cargando datos...\n")

datos = cargar_datos()

print(f"✓ Train: {datos.entrenamiento.shape}")
print(f"✓ Test: {datos.prueba.shape}")

# Lista de verificaciones que se van acumulando en todo el script.
# Cada elemento es (descripcion, True/False). Al final se revisan todas
# y se guardan en un reporte: si alguna falla, el script se detiene aquí
# mismo en vez de seguir entrenando con datos que podrían estar mal.
verificaciones = []


def verificar(descripcion, condicion):
    verificaciones.append((descripcion, bool(condicion)))
    simbolo = "✓" if condicion else "✗"
    print(f"  {simbolo} {descripcion}")
    return condicion


print("\nVerificando consistencia de los datos cargados:")
verificar("Train: mismo número de filas en datos y actividades",
          len(datos.entrenamiento) == len(datos.actividades_entrenamiento))
verificar("Test: mismo número de filas en datos y actividades",
          len(datos.prueba) == len(datos.actividades_prueba))
verificar("Train y test tienen el mismo número de características",
          datos.entrenamiento.shape[1] == datos.prueba.shape[1])
verificar("Las 6 actividades esperadas están presentes en train",
          set(np.unique(datos.actividades_entrenamiento)) == set(datos.nombres_actividades.keys()))
verificar("El dataset no tiene valores faltantes tras la limpieza",
          datos.entrenamiento.isnull().sum().sum() == 0 and datos.prueba.isnull().sum().sum() == 0)

# 2. NORMALIZACIÓN
print("\n2️⃣  Normalizando datos...\n")

normalizador = StandardScaler()
entrenamiento_normalizado = normalizador.fit_transform(datos.entrenamiento)
prueba_normalizada = normalizador.transform(datos.prueba)

# Se guarda el normalizador ya ajustado para que el script de evaluación
# use exactamente el mismo, en vez de tener que recalcularlo.
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
joblib.dump(normalizador, RUTA_NORMALIZADOR)

print(f"✓ Datos normalizados")
print(f"  - Media train: {entrenamiento_normalizado.mean():.4f}")
print(f"  - Std train: {entrenamiento_normalizado.std():.4f}")
print(f"  - Media test: {prueba_normalizada.mean():.4f}")
print(f"  - Std test: {prueba_normalizada.std():.4f}")
print(f"✓ Normalizador guardado en: {RUTA_NORMALIZADOR}")

verificar("La normalización no generó valores NaN (train)",
          not np.isnan(entrenamiento_normalizado).any())
verificar("La normalización no generó valores NaN (test)",
          not np.isnan(prueba_normalizada).any())
verificar("Media de train normalizado es aproximadamente 0",
          abs(entrenamiento_normalizado.mean()) < 0.01)
verificar("Desviación estándar de train normalizado es aproximadamente 1",
          abs(entrenamiento_normalizado.std() - 1) < 0.01)
verificar("El normalizador se guardó en disco",
          RUTA_NORMALIZADOR.exists() and RUTA_NORMALIZADOR.stat().st_size > 0)

# Verificar que las 6 actividades estén balanceadas entre sí
print(f"\n✓ Distribución de clases:")
for id_actividad in sorted(datos.nombres_actividades.keys()):
    cantidad = (datos.actividades_entrenamiento == id_actividad).sum()
    porcentaje = (cantidad / len(datos.actividades_entrenamiento)) * 100
    print(f"  {datos.nombres_actividades[id_actividad]:20} → {cantidad:4} ({porcentaje:5.1f}%)")

# 3. CREAR Y ENTRENAR MODELOS
print("\n3️⃣  Creando modelos...\n")

entrenador = EntrenadorModelos()
entrenador.crear_modelos()

print("\n4️⃣  Entrenando modelos...\n")

CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)

resultados_por_modelo = {}

for nombre_modelo in entrenador.modelos.keys():
    print(f"\n{'='*50}")
    print(f"Entrenando: {nombre_modelo}")
    print(f"{'='*50}")

    momento_inicio = time.time()
    entrenador.entrenar_modelo(nombre_modelo, entrenamiento_normalizado, datos.actividades_entrenamiento)
    momento_fin = time.time()

    tiempo_entrenamiento = momento_fin - momento_inicio
    print(f"⏱️  Tiempo: {tiempo_entrenamiento:.2f}s")

    # Guardar modelo en disco
    ruta_modelo = CARPETA_MODELOS / f"{nombre_modelo.replace(' ', '_')}.pkl"
    entrenador.guardar_modelo(nombre_modelo, ruta_modelo)
    verificar(f"{nombre_modelo}: el archivo .pkl se guardó en disco",
              ruta_modelo.exists() and ruta_modelo.stat().st_size > 0)

    # Medir exactitud en train y en test
    predicciones_train = entrenador.predecir(nombre_modelo, entrenamiento_normalizado)
    exactitud_train = (predicciones_train == datos.actividades_entrenamiento).mean()

    predicciones_test = entrenador.predecir(nombre_modelo, prueba_normalizada)
    exactitud_test = (predicciones_test == datos.actividades_prueba).mean()

    print(f"✓ Train Accuracy: {exactitud_train:.4f}")
    print(f"✓ Test Accuracy:  {exactitud_test:.4f}")

    verificar(f"{nombre_modelo}: predicciones dentro de las actividades válidas (1-6)",
              set(np.unique(predicciones_test)).issubset(set(datos.nombres_actividades.keys())))
    verificar(f"{nombre_modelo}: exactitud de test dentro del rango [0, 1]",
              0.0 <= exactitud_test <= 1.0)

    resultados_por_modelo[nombre_modelo] = {
        'exactitud_train': exactitud_train,
        'exactitud_test': exactitud_test,
        'tiempo': tiempo_entrenamiento
    }

# 5. VERIFICACIÓN FINAL
print("\n5️⃣  Verificación final:\n")

verificar("Se entrenaron y guardaron los 5 modelos esperados",
          len(resultados_por_modelo) == 5)

ruta_verificacion = CARPETA_RESULTADOS / "verificacion_entrenamiento.txt"
with open(ruta_verificacion, 'w', encoding='utf-8') as archivo:
    archivo.write("VERIFICACIÓN DEL ENTRENAMIENTO\n")
    archivo.write("=" * 60 + "\n\n")
    for descripcion, paso in verificaciones:
        archivo.write(f"[{'OK' if paso else 'FALLO'}] {descripcion}\n")
    archivo.write(f"\nTotal: {sum(p for _, p in verificaciones)}/{len(verificaciones)} verificaciones pasaron.\n")

print(f"✓ Reporte de verificación guardado en: {ruta_verificacion}")

if not all(paso for _, paso in verificaciones):
    fallidas = [descripcion for descripcion, paso in verificaciones if not paso]
    raise AssertionError(
        "El entrenamiento no pasó todas las verificaciones:\n  - " + "\n  - ".join(fallidas)
    )

# 6. RESUMEN
print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)

print("\n📊 RESUMEN DE MODELOS:\n")
print(f"{'Modelo':<25} {'Train Acc':<12} {'Test Acc':<12} {'Tiempo (s)':<12}")
print("-" * 61)

for nombre_modelo in sorted(resultados_por_modelo.keys()):
    resultado = resultados_por_modelo[nombre_modelo]
    print(f"{nombre_modelo:<25} {resultado['exactitud_train']:<12.4f} "
          f"{resultado['exactitud_test']:<12.4f} {resultado['tiempo']:<12.2f}")

# Encontrar el modelo con mejor exactitud en test
nombre_mejor_modelo, datos_mejor_modelo = max(
    resultados_por_modelo.items(), key=lambda item: item[1]['exactitud_test']
)
print(f"\n🏆 Mejor modelo: {nombre_mejor_modelo}")
print(f"   Test Accuracy: {datos_mejor_modelo['exactitud_test']:.4f}")

# Guardar resumen en un archivo de texto
ruta_resumen = CARPETA_RESULTADOS / "entrenamiento_summary.txt"
with open(ruta_resumen, 'w', encoding='utf-8') as archivo:
    archivo.write("RESUMEN DE ENTRENAMIENTO\n")
    archivo.write("=" * 70 + "\n\n")
    archivo.write(f"{'Modelo':<25} {'Train Acc':<12} {'Test Acc':<12} {'Tiempo (s)':<12}\n")
    archivo.write("-" * 61 + "\n")
    for nombre_modelo in sorted(resultados_por_modelo.keys()):
        resultado = resultados_por_modelo[nombre_modelo]
        archivo.write(f"{nombre_modelo:<25} {resultado['exactitud_train']:<12.4f} "
                       f"{resultado['exactitud_test']:<12.4f} {resultado['tiempo']:<12.2f}\n")

print(f"\n📁 Modelos guardados en: {CARPETA_MODELOS}")
print(f"📄 Resumen guardado en: {ruta_resumen}")

print(f"\n🎯 Próximo paso: python scripts/03_evaluacion.py")
print("="*70 + "\n")
