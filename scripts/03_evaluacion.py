"""
Proyecto: Reconocimiento de Actividad Humana (HAR)
Script: 03_evaluacion.py
Descripción: Evaluación y Comparación de Modelos
"""

import numpy as np
import sys
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Fuerza salida UTF-8: algunas consolas Windows usan cp1252/cp850 por
# defecto y no pueden imprimir los caracteres ✓/📊/🎯 usados en este script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Agregar la carpeta del proyecto al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datos import cargar_datos, CARPETA_MODELOS, CARPETA_RESULTADOS, CARPETA_GRAFICOS, RUTA_NORMALIZADOR
from src.models import EntrenadorModelos
from src.evaluation import EvaluadorModelos, NOMBRES_METRICAS_ES

print("\n" + "="*70)
print("📊 EVALUACIÓN DE MODELOS - PROYECTO HAR")
print("="*70 + "\n")

# 1. CARGAR DATOS
print("1️⃣  Cargando datos...\n")

# Se cargan train y test juntos porque la limpieza del test puede depender
# de estadísticas del train (por ejemplo, la media usada para imputar).
datos = cargar_datos()

# Lista de verificaciones que se van acumulando en todo el script.
# Al final se revisan todas y se guardan en un reporte: si alguna falla,
# el script se detiene en vez de dar por buenos resultados incorrectos.
verificaciones = []


def verificar(descripcion, condicion):
    verificaciones.append((descripcion, bool(condicion)))
    simbolo = "✓" if condicion else "✗"
    print(f"  {simbolo} {descripcion}")
    return condicion


print("Verificando datos de prueba:")
verificar("Las 6 actividades esperadas están presentes en test",
          set(np.unique(datos.actividades_prueba)) == set(datos.nombres_actividades.keys()))
verificar("El normalizador del entrenamiento existe en disco",
          RUTA_NORMALIZADOR.exists())

if not RUTA_NORMALIZADOR.exists():
    raise FileNotFoundError(
        f"No se encontró {RUTA_NORMALIZADOR}.\n"
        "Ejecuta primero: python scripts/02_entrenamiento.py"
    )

# Se reutiliza el normalizador exacto que se usó al entrenar, en vez de
# volver a calcularlo: así no hay forma de que difiera del original.
normalizador = joblib.load(RUTA_NORMALIZADOR)
prueba_normalizada = normalizador.transform(datos.prueba)

print(f"\n✓ Datos cargados y normalizados con el normalizador del entrenamiento")

verificar("La normalización de test no generó valores NaN",
          not np.isnan(prueba_normalizada).any())

# 2. CARGAR MODELOS ENTRENADOS
print("\n2️⃣  Cargando modelos entrenados...\n")

entrenador = EntrenadorModelos()

archivos_modelos = list(CARPETA_MODELOS.glob("*.pkl"))
print(f"✓ Encontrados {len(archivos_modelos)} modelos\n")

for archivo_modelo in archivos_modelos:
    nombre_modelo = archivo_modelo.stem.replace('_', ' ')
    entrenador.cargar_modelo(nombre_modelo, archivo_modelo)
    print(f"  ✓ {nombre_modelo}")

verificar("Se encontraron y cargaron los 5 modelos esperados", len(archivos_modelos) == 5)

# 3. EVALUAR MODELOS
print("\n3️⃣  Evaluando modelos...\n")

evaluador = EvaluadorModelos()

for nombre_modelo in entrenador.modelos_entrenados.keys():
    print(f"\nEvaluando: {nombre_modelo}")
    print("-" * 50)

    predicciones_test = entrenador.predecir(nombre_modelo, prueba_normalizada)
    probabilidades = entrenador.predecir_probabilidad(nombre_modelo, prueba_normalizada)

    evaluador.evaluar_modelo(nombre_modelo, datos.actividades_prueba, predicciones_test, probabilidades)

    metricas = evaluador.resultados[nombre_modelo]
    print(f"  Accuracy:  {metricas.get('Accuracy', 0):.4f}")
    print(f"  Precision: {metricas.get('Precision', 0):.4f}")
    print(f"  Recall:    {metricas.get('Recall', 0):.4f}")
    print(f"  F1-Score:  {metricas.get('F1-Score', 0):.4f}")

    verificar(f"{nombre_modelo}: todas las métricas están dentro del rango [0, 1]",
              all(0.0 <= metricas[m] <= 1.0 for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']))

# 4. COMPARAR MODELOS
print("\n4️⃣  Comparando modelos...\n")

tabla_comparacion = evaluador.comparar_modelos()

# 5. GRÁFICOS
print("\n5️⃣  Generando gráficos...\n")

CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# Gráfico 1: Comparación de métricas
fig, eje = plt.subplots(figsize=(12, 6))

tabla_metricas = tabla_comparacion[['Accuracy', 'Precision', 'Recall', 'F1-Score']].sort_values('F1-Score', ascending=False)
tabla_metricas.rename(columns=NOMBRES_METRICAS_ES).plot(kind='bar', ax=eje, width=0.8)
eje.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
eje.set_ylabel('Puntaje')
eje.set_xlabel('Modelo')
eje.set_ylim([0, 1.05])
eje.legend(loc='lower right')
eje.grid(axis='y', alpha=0.3)
plt.setp(eje.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "06_comparacion_modelos.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 06_comparacion_modelos.png")
plt.close()

# Gráfico 2: F1-Score por modelo
fig, eje = plt.subplots(figsize=(10, 6))

puntajes_f1 = tabla_comparacion.sort_values('F1-Score', ascending=True)['F1-Score']
colores_barras = ['green' if nombre == puntajes_f1.idxmax() else 'steelblue' for nombre in puntajes_f1.index]
eje.barh(range(len(puntajes_f1)), puntajes_f1.values, color=colores_barras, alpha=0.8)
eje.set_yticks(range(len(puntajes_f1)))
eje.set_yticklabels(puntajes_f1.index)
eje.set_xlabel('Puntaje F1')
eje.set_title('Ranking de Modelos (Puntaje F1)', fontsize=14, fontweight='bold')
eje.set_xlim([0, 1])
eje.grid(axis='x', alpha=0.3)

for posicion, valor in enumerate(puntajes_f1.values):
    eje.text(valor + 0.02, posicion, f'{valor:.4f}', va='center')

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "07_ranking_modelos.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 07_ranking_modelos.png")
plt.close()

# Gráfico 3: Matrices de confusión para los 3 mejores modelos
mejores_modelos = tabla_comparacion.nlargest(3, 'F1-Score').index

fig, ejes = plt.subplots(1, 3, figsize=(18, 5))

for posicion, nombre_modelo in enumerate(mejores_modelos):
    predicciones = entrenador.predecir(nombre_modelo, prueba_normalizada)
    matriz_confusion = confusion_matrix(datos.actividades_prueba, predicciones)

    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=ejes[posicion],
                cbar=False, xticklabels=range(1, 7), yticklabels=range(1, 7))
    ejes[posicion].set_title(f'{nombre_modelo}\nPuntaje F1: {tabla_comparacion.loc[nombre_modelo, "F1-Score"]:.4f}')
    ejes[posicion].set_ylabel('Verdadero')
    ejes[posicion].set_xlabel('Predicción')

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "08_confusion_matrices.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 08_confusion_matrices.png")
plt.close()

# Gráfico 4: Radar chart del mejor modelo
nombre_mejor_modelo = tabla_comparacion['F1-Score'].idxmax()
metricas_mejor_modelo = tabla_comparacion.loc[nombre_mejor_modelo, ['Accuracy', 'Precision', 'Recall', 'F1-Score']]

fig, eje = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

angulos = np.linspace(0, 2 * np.pi, len(metricas_mejor_modelo), endpoint=False).tolist()
valores = metricas_mejor_modelo.values.tolist()

angulos += angulos[:1]
valores += valores[:1]

eje.plot(angulos, valores, 'o-', linewidth=2, color='green', label=nombre_mejor_modelo)
eje.fill(angulos, valores, alpha=0.25, color='green')
eje.set_xticks(angulos[:-1])
eje.set_xticklabels([NOMBRES_METRICAS_ES[nombre] for nombre in metricas_mejor_modelo.index])
eje.set_ylim(0, 1)
eje.set_title(f'Métricas del Mejor Modelo\n{nombre_mejor_modelo}', fontsize=14, fontweight='bold', pad=20)
eje.grid(True)

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "09_radar_mejor_modelo.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 09_radar_mejor_modelo.png")
plt.close()

# 6. GUARDAR RESULTADOS EN CSV
print("\n6️⃣  Guardando resultados...\n")

ruta_csv = CARPETA_RESULTADOS / "metricas_modelos.csv"
tabla_comparacion.to_csv(ruta_csv, index_label='Modelo')
print(f"✓ Resultados guardados en: {ruta_csv}")

# 7. REPORTE DETALLADO
print("\n7️⃣  Generando reporte detallado...\n")

predicciones_mejor_modelo = entrenador.predecir(nombre_mejor_modelo, prueba_normalizada)

ruta_reporte = CARPETA_RESULTADOS / "reporte_detallado.txt"

with open(ruta_reporte, 'w', encoding='utf-8') as archivo:
    archivo.write("="*70 + "\n")
    archivo.write("REPORTE DE EVALUACIÓN - PROYECTO HAR\n")
    archivo.write("="*70 + "\n\n")

    archivo.write("COMPARACIÓN DE MODELOS\n")
    archivo.write("-"*70 + "\n")
    archivo.write(tabla_comparacion.to_string())
    archivo.write("\n\n")

    archivo.write("MEJOR MODELO\n")
    archivo.write("-"*70 + "\n")
    archivo.write(f"Nombre: {nombre_mejor_modelo}\n")
    archivo.write(f"F1-Score: {tabla_comparacion.loc[nombre_mejor_modelo, 'F1-Score']:.4f}\n")
    archivo.write(f"Accuracy: {tabla_comparacion.loc[nombre_mejor_modelo, 'Accuracy']:.4f}\n")
    archivo.write(f"Precision: {tabla_comparacion.loc[nombre_mejor_modelo, 'Precision']:.4f}\n")
    archivo.write(f"Recall: {tabla_comparacion.loc[nombre_mejor_modelo, 'Recall']:.4f}\n\n")

    archivo.write("CLASIFICACIÓN POR ACTIVIDAD (Mejor Modelo)\n")
    archivo.write("-"*70 + "\n")
    archivo.write(classification_report(
        datos.actividades_prueba, predicciones_mejor_modelo,
        target_names=[datos.nombres_actividades[i] for i in sorted(datos.nombres_actividades.keys())]
    ))

print(f"✓ Reporte guardado en: {ruta_reporte}")

# 8. VERIFICACIÓN FINAL
print("\n8️⃣  Verificación final:\n")

archivos_esperados = [
    CARPETA_GRAFICOS / "06_comparacion_modelos.png",
    CARPETA_GRAFICOS / "07_ranking_modelos.png",
    CARPETA_GRAFICOS / "08_confusion_matrices.png",
    CARPETA_GRAFICOS / "09_radar_mejor_modelo.png",
    ruta_csv,
    ruta_reporte,
]
for ruta_archivo in archivos_esperados:
    verificar(f"El archivo {ruta_archivo.name} se generó correctamente",
              ruta_archivo.exists() and ruta_archivo.stat().st_size > 0)

ruta_verificacion = CARPETA_RESULTADOS / "verificacion_evaluacion.txt"
with open(ruta_verificacion, 'w', encoding='utf-8') as archivo:
    archivo.write("VERIFICACIÓN DE LA EVALUACIÓN\n")
    archivo.write("=" * 60 + "\n\n")
    for descripcion, paso in verificaciones:
        archivo.write(f"[{'OK' if paso else 'FALLO'}] {descripcion}\n")
    archivo.write(f"\nTotal: {sum(p for _, p in verificaciones)}/{len(verificaciones)} verificaciones pasaron.\n")

print(f"✓ Reporte de verificación guardado en: {ruta_verificacion}")

if not all(paso for _, paso in verificaciones):
    fallidas = [descripcion for descripcion, paso in verificaciones if not paso]
    raise AssertionError(
        "La evaluación no pasó todas las verificaciones:\n  - " + "\n  - ".join(fallidas)
    )

# 9. RESUMEN FINAL
print("\n" + "="*70)
print("✅ EVALUACIÓN COMPLETADA")
print("="*70)

print(f"\n🏆 MEJOR MODELO: {nombre_mejor_modelo}")
print(f"   Accuracy:  {tabla_comparacion.loc[nombre_mejor_modelo, 'Accuracy']:.4f}")
print(f"   F1-Score:  {tabla_comparacion.loc[nombre_mejor_modelo, 'F1-Score']:.4f}")

print(f"\n📁 Gráficos guardados en: {CARPETA_GRAFICOS}")
print(f"📄 Resultados CSV: {ruta_csv}")
print(f"📋 Reporte: {ruta_reporte}")

print(f"\n📊 Archivos generados:")
print(f"   • 06_comparacion_modelos.png")
print(f"   • 07_ranking_modelos.png")
print(f"   • 08_confusion_matrices.png")
print(f"   • 09_radar_mejor_modelo.png")
print(f"   • metricas_modelos.csv")
print(f"   • reporte_detallado.txt")

print(f"\n🎯 Próximos pasos:")
print(f"   1. Revisar gráficos en results/graficos/")
print(f"   2. Leer reporte en results/reporte_detallado.txt")
print(f"   3. Preparar presentación con los resultados")

print("="*70 + "\n")
