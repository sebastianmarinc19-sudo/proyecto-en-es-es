"""
Proyecto: Reconocimiento de Actividad Humana (HAR)
Script: 04_prediccion.py
Descripción: Demo — toma muestras del conjunto de prueba y predice su actividad
             con el mejor modelo entrenado, mostrando si aciertó y con qué confianza.

Uso:
    python scripts/04_prediccion.py           → 5 muestras al azar
    python scripts/04_prediccion.py 10        → 10 muestras al azar
    python scripts/04_prediccion.py --indice 42   → una muestra específica (para
                                                     repetir el mismo ejemplo en vivo)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Fuerza salida UTF-8: algunas consolas Windows usan cp1252/cp850 por
# defecto y no pueden imprimir los caracteres ✓/📊/🎯 usados en este script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Agregar la carpeta del proyecto al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datos import cargar_datos, CARPETA_MODELOS, CARPETA_RESULTADOS, RUTA_NORMALIZADOR, NOMBRES_ACTIVIDADES_ES
from src.models import EntrenadorModelos

print("\n" + "="*70)
print("🔮 DEMO DE PREDICCIÓN - PROYECTO HAR")
print("="*70 + "\n")

# 0. LEER ARGUMENTOS
parser = argparse.ArgumentParser(description="Demo de predicción con el mejor modelo entrenado")
parser.add_argument("cantidad", type=int, nargs="?", default=5,
                     help="Cuántas muestras al azar mostrar (por defecto 5)")
parser.add_argument("--indice", type=int, default=None,
                     help="Mostrar una muestra específica del conjunto de prueba (0 a 2946)")
parser.add_argument("--semilla", type=int, default=42,
                     help="Semilla aleatoria, para repetir el mismo grupo de muestras (por defecto 42)")
args = parser.parse_args()

# Lista de verificaciones: si algo falla, el script se detiene en vez de
# mostrar una "demo" con resultados que no se pueden confiar.
verificaciones = []


def verificar(descripcion, condicion):
    verificaciones.append((descripcion, bool(condicion)))
    simbolo = "✓" if condicion else "✗"
    print(f"  {simbolo} {descripcion}")
    return condicion


# 1. ENCONTRAR EL MEJOR MODELO
print("1️⃣  Buscando el mejor modelo entrenado...\n")

ruta_metricas = CARPETA_RESULTADOS / "metricas_modelos.csv"
verificar("Existe results/metricas_modelos.csv (requiere haber corrido 03_evaluacion.py)",
          ruta_metricas.exists())

if not ruta_metricas.exists():
    raise FileNotFoundError(
        f"No se encontró {ruta_metricas}.\n"
        "Ejecuta primero: python scripts/02_entrenamiento.py y python scripts/03_evaluacion.py"
    )

tabla_metricas = pd.read_csv(ruta_metricas, index_col=0)
nombre_mejor_modelo = tabla_metricas["F1-Score"].idxmax()
puntaje_mejor_modelo = tabla_metricas.loc[nombre_mejor_modelo, "F1-Score"]

print(f"✓ Mejor modelo: {nombre_mejor_modelo} (Puntaje F1: {puntaje_mejor_modelo:.4f})")

# 2. CARGAR MODELO, NORMALIZADOR Y DATOS
print("\n2️⃣  Cargando modelo, normalizador y datos...\n")

verificar("El normalizador existe en disco", RUTA_NORMALIZADOR.exists())

ruta_modelo = CARPETA_MODELOS / f"{nombre_mejor_modelo.replace(' ', '_')}.pkl"
verificar(f"El archivo del modelo '{nombre_mejor_modelo}' existe en disco", ruta_modelo.exists())

normalizador = joblib.load(RUTA_NORMALIZADOR)

entrenador = EntrenadorModelos()
entrenador.cargar_modelo(nombre_mejor_modelo, ruta_modelo)

datos = cargar_datos()
prueba_normalizada = normalizador.transform(datos.prueba)

print(f"✓ Listo: {len(datos.prueba)} muestras de prueba disponibles para la demo")

# 3. ELEGIR LAS MUESTRAS A PREDECIR
print("\n3️⃣  Preparando muestras...\n")

total_muestras = len(datos.prueba)

if args.indice is not None:
    indice_valido = verificar(f"El índice {args.indice} está dentro del rango [0, {total_muestras - 1}]",
                               0 <= args.indice < total_muestras)
    if not indice_valido:
        raise ValueError(
            f"--indice {args.indice} está fuera de rango. "
            f"Usa un valor entre 0 y {total_muestras - 1}."
        )
    indices = [args.indice]
else:
    cantidad = max(1, min(args.cantidad, total_muestras))
    generador_aleatorio = np.random.RandomState(args.semilla)
    indices = generador_aleatorio.choice(total_muestras, size=cantidad, replace=False)

print(f"✓ Se van a predecir {len(indices)} muestra(s)")

# 4. PREDECIR Y MOSTRAR RESULTADOS
print("\n4️⃣  Resultados de la predicción:\n")
print(f"{'#':<4} {'Actividad Real':<18} {'Predicción':<18} {'Confianza':<12} {'¿Acertó?'}")
print("-" * 70)

aciertos = 0
filas_reporte = []

for indice in indices:
    muestra = prueba_normalizada[indice].reshape(1, -1)
    actividad_real_id = datos.actividades_prueba[indice]

    prediccion_id = entrenador.predecir(nombre_mejor_modelo, muestra)[0]
    probabilidades = entrenador.predecir_probabilidad(nombre_mejor_modelo, muestra)
    confianza = probabilidades[0].max() if probabilidades is not None else float("nan")

    nombre_real = NOMBRES_ACTIVIDADES_ES[actividad_real_id]
    nombre_prediccion = NOMBRES_ACTIVIDADES_ES[prediccion_id]
    acerto = actividad_real_id == prediccion_id
    aciertos += int(acerto)

    simbolo_acierto = "✓ Sí" if acerto else "✗ No"
    print(f"{indice:<4} {nombre_real:<18} {nombre_prediccion:<18} {confianza*100:>6.1f}%     {simbolo_acierto}")

    filas_reporte.append({
        "indice": int(indice),
        "actividad_real": nombre_real,
        "prediccion": nombre_prediccion,
        "confianza": round(float(confianza) * 100, 1),
        "acerto": acerto,
    })

exactitud_demo = aciertos / len(indices)
print("-" * 70)
print(f"Exactitud en esta demo: {aciertos}/{len(indices)} ({exactitud_demo*100:.1f}%)")

verificar("Todas las predicciones son actividades válidas (1-6)",
          all(fila["prediccion"] in NOMBRES_ACTIVIDADES_ES.values() for fila in filas_reporte))

# 5. GUARDAR REPORTE
print("\n5️⃣  Guardando reporte...\n")

ruta_reporte = CARPETA_RESULTADOS / "demo_predicciones.txt"
with open(ruta_reporte, "w", encoding="utf-8") as archivo:
    archivo.write("DEMO DE PREDICCIÓN\n")
    archivo.write("=" * 60 + "\n\n")
    archivo.write(f"Modelo usado: {nombre_mejor_modelo} (Puntaje F1: {puntaje_mejor_modelo:.4f})\n\n")
    archivo.write(f"{'#':<6}{'Actividad Real':<18}{'Predicción':<18}{'Confianza':<12}{'Acertó'}\n")
    for fila in filas_reporte:
        archivo.write(f"{fila['indice']:<6}{fila['actividad_real']:<18}{fila['prediccion']:<18}"
                       f"{fila['confianza']:>6.1f}%     {'Si' if fila['acerto'] else 'No'}\n")
    archivo.write(f"\nExactitud en esta demo: {aciertos}/{len(indices)} ({exactitud_demo*100:.1f}%)\n")

print(f"✓ Reporte guardado en: {ruta_reporte}")

# 6. RESUMEN FINAL
print("\n" + "="*70)
print("✅ DEMO COMPLETADA" if all(p for _, p in verificaciones) else "⚠ DEMO CON PROBLEMAS")
print("="*70)

if not all(paso for _, paso in verificaciones):
    fallidas = [descripcion for descripcion, paso in verificaciones if not paso]
    raise AssertionError(
        "La demo no pasó todas las verificaciones:\n  - " + "\n  - ".join(fallidas)
    )

print(f"\n🏆 Modelo usado: {nombre_mejor_modelo}")
print(f"📊 Exactitud en esta demo: {exactitud_demo*100:.1f}% ({aciertos}/{len(indices)})")
print(f"📄 Reporte: {ruta_reporte}")
print(f"\n💡 Tip: para repetir un ejemplo exacto en la presentación, usa:")
print(f"   python scripts/04_prediccion.py --indice {indices[0]}")
print("="*70 + "\n")
