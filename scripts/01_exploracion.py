"""
Proyecto: Reconocimiento de Actividad Humana (HAR)
Script: 01_exploracion.py
Descripción: Análisis Exploratorio de Datos (EDA)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Fuerza salida UTF-8: algunas consolas Windows usan cp1252/cp850 por
# defecto y no pueden imprimir los caracteres ✓/📊/🎯 usados en este script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Agregar la carpeta del proyecto al path para poder importar src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datos import cargar_datos, guardar_reporte_limpieza, CARPETA_GRAFICOS

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*70)
print("📊 ANÁLISIS EXPLORATORIO DE DATOS - PROYECTO HAR")
print("="*70 + "\n")

# 1. CARGAR DATOS
print("1️⃣  Cargando datos...\n")

datos = cargar_datos()

print(f"✓ Train: {datos.entrenamiento.shape}")
print(f"✓ Test: {datos.prueba.shape}")
print(f"✓ Características: {len(datos.nombres_caracteristicas)}")
print(f"✓ Actividades: {len(datos.nombres_actividades)}\n")

# 2. INFORMACIÓN GENERAL
print("2️⃣  Información del Dataset:\n")

print("Actividades:")
for id_actividad, nombre_actividad in datos.nombres_actividades.items():
    cantidad_train = (datos.actividades_entrenamiento == id_actividad).sum()
    cantidad_test = (datos.actividades_prueba == id_actividad).sum()
    print(f"   {nombre_actividad:20} → Train: {cantidad_train:4} | Test: {cantidad_test:4}")

print(f"\nVoluntarios:")
print(f"  - Train: {len(np.unique(datos.voluntarios_entrenamiento))} personas")
print(f"  - Test: {len(np.unique(datos.voluntarios_prueba))} personas")

print(f"\nTotal muestras:")
print(f"  - Train: {len(datos.entrenamiento)}")
print(f"  - Test: {len(datos.prueba)}")
print(f"  - Total: {len(datos.entrenamiento) + len(datos.prueba)}")

# 3. LIMPIEZA Y VERIFICACIÓN DE DATOS
# La limpieza ya se aplicó dentro de cargar_datos(), así que estos mismos
# datos limpios son los que usan también los scripts 02 y 03.
print("\n3️⃣  Limpieza y Verificación de Datos:\n")

reporte = datos.reporte_limpieza

print(f"Valores faltantes (train): {reporte['faltantes_train']}")
print(f"Valores faltantes (test):  {reporte['faltantes_test']}")
print(f"Filas duplicadas (train):  {reporte['duplicados_train']}")
print(f"Filas duplicadas (test):   {reporte['duplicados_test']}")
print(f"Etiquetas válidas (train): {'Sí' if reporte['etiquetas_train_validas'] else 'No'}")
print(f"Etiquetas válidas (test):  {'Sí' if reporte['etiquetas_test_validas'] else 'No'}")
print(f"Rango de valores: [{reporte['valor_minimo']:.4f}, {reporte['valor_maximo']:.4f}]")

if reporte['dataset_estaba_limpio']:
    print("\n✓ No se encontraron valores faltantes ni filas duplicadas.")
    print("  El dataset ya viene normalizado por el proveedor (UCI), no requiere limpieza adicional.")
else:
    print("")
    for accion in reporte['acciones']:
        print(f"⚠ {accion}")

ruta_reporte_limpieza = guardar_reporte_limpieza(reporte)
print(f"✓ Reporte de limpieza guardado en: {ruta_reporte_limpieza}")

# 4. ESTADÍSTICAS DESCRIPTIVAS
print("\n4️⃣  Estadísticas Descriptivas:\n")

print("Train - Primeros valores de algunas características:")
print(f"Mean: {datos.entrenamiento.mean().mean():.4f}")
print(f"Std:  {datos.entrenamiento.std().mean():.4f}")
print(f"Min:  {datos.entrenamiento.min().min():.4f}")
print(f"Max:  {datos.entrenamiento.max().max():.4f}")

# 5. GRÁFICOS
print("\n5️⃣  Generando gráficos...\n")

CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# Gráfico 1: Distribución de actividades
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

lista_actividades = datos.lista_actividades_en_espanol()
ids_actividades = sorted(datos.nombres_actividades.keys())
conteo_train = [np.sum(datos.actividades_entrenamiento == i) for i in ids_actividades]
conteo_test = [np.sum(datos.actividades_prueba == i) for i in ids_actividades]

posiciones = np.arange(len(lista_actividades))
ancho_barra = 0.35

ejes[0].bar(posiciones - ancho_barra/2, conteo_train, ancho_barra, label='Entrenamiento', alpha=0.8)
ejes[0].bar(posiciones + ancho_barra/2, conteo_test, ancho_barra, label='Prueba', alpha=0.8)
ejes[0].set_xlabel('Actividad')
ejes[0].set_ylabel('Número de Muestras')
ejes[0].set_title('Distribución de Actividades')
ejes[0].set_xticks(posiciones)
ejes[0].set_xticklabels(lista_actividades, rotation=45, ha='right')
ejes[0].legend()
ejes[0].grid(axis='y', alpha=0.3)

# Gráfico 2: Balance de datos
total_por_actividad = [conteo_train[i] + conteo_test[i] for i in range(len(lista_actividades))]
ejes[1].pie(total_por_actividad, labels=lista_actividades, autopct='%1.1f%%', startangle=90)
ejes[1].set_title('Proporción de Actividades')

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "01_distribucion_actividades.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_distribucion_actividades.png")
plt.close()

# Gráfico 3: Distribución de voluntarios
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

ids_train, cantidad_por_id_train = np.unique(datos.voluntarios_entrenamiento, return_counts=True)
ids_test, cantidad_por_id_test = np.unique(datos.voluntarios_prueba, return_counts=True)

ejes[0].bar(ids_train, cantidad_por_id_train, alpha=0.7, color='steelblue')
ejes[0].set_xlabel('ID Voluntario')
ejes[0].set_ylabel('Número de Muestras')
ejes[0].set_title('Distribución por Voluntario - Entrenamiento')
ejes[0].grid(axis='y', alpha=0.3)

ejes[1].bar(ids_test, cantidad_por_id_test, alpha=0.7, color='coral')
ejes[1].set_xlabel('ID Voluntario')
ejes[1].set_ylabel('Número de Muestras')
ejes[1].set_title('Distribución por Voluntario - Prueba')
ejes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "02_distribucion_voluntarios.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_distribucion_voluntarios.png")
plt.close()

# Gráfico 4: Estadísticas de características
fig, ejes = plt.subplots(2, 2, figsize=(14, 10))

ejes[0, 0].hist(datos.entrenamiento.mean(), bins=30, alpha=0.7, color='steelblue')
ejes[0, 0].set_xlabel('Media')
ejes[0, 0].set_ylabel('Frecuencia')
ejes[0, 0].set_title('Distribución de Medias por Característica')
ejes[0, 0].grid(alpha=0.3)

ejes[0, 1].hist(datos.entrenamiento.std(), bins=30, alpha=0.7, color='coral')
ejes[0, 1].set_xlabel('Desviación Estándar')
ejes[0, 1].set_ylabel('Frecuencia')
ejes[0, 1].set_title('Distribución de Desviación Estándar por Característica')
ejes[0, 1].grid(alpha=0.3)

ejes[1, 0].hist(datos.entrenamiento.min(), bins=30, alpha=0.7, color='green')
ejes[1, 0].set_xlabel('Valor Mínimo')
ejes[1, 0].set_ylabel('Frecuencia')
ejes[1, 0].set_title('Distribución de Mínimos por Característica')
ejes[1, 0].grid(alpha=0.3)

ejes[1, 1].hist(datos.entrenamiento.max(), bins=30, alpha=0.7, color='orange')
ejes[1, 1].set_xlabel('Valor Máximo')
ejes[1, 1].set_ylabel('Frecuencia')
ejes[1, 1].set_title('Distribución de Máximos por Característica')
ejes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "03_estadisticas_caracteristicas.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 03_estadisticas_caracteristicas.png")
plt.close()

# Gráfico 5: Boxplot de algunas características seleccionadas
fig, ejes = plt.subplots(2, 3, figsize=(16, 10))
ejes = ejes.flatten()

caracteristicas_a_revisar = [0, 50, 100, 200, 300, 400]
for posicion, columna in enumerate(caracteristicas_a_revisar):
    valores_por_actividad = [datos.entrenamiento[datos.actividades_entrenamiento == i][columna]
                              for i in ids_actividades]
    ejes[posicion].boxplot(valores_por_actividad, labels=lista_actividades)
    ejes[posicion].set_title(f'Característica {columna}')
    ejes[posicion].set_ylabel('Valor')
    ejes[posicion].grid(alpha=0.3)
    plt.setp(ejes[posicion].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "04_boxplot_caracteristicas.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 04_boxplot_caracteristicas.png")
plt.close()

# Gráfico 6: Correlación de primeras características
print("✓ Procesando correlaciones...")
primeras_20_caracteristicas = datos.entrenamiento.iloc[:, :20]
matriz_correlacion = primeras_20_caracteristicas.corr()

fig, eje = plt.subplots(figsize=(12, 10))
sns.heatmap(matriz_correlacion, cmap='coolwarm', center=0, ax=eje,
            cbar_kws={'label': 'Correlación'})
eje.set_title('Matriz de Correlación (Primeras 20 Características)')
plt.tight_layout()
plt.savefig(CARPETA_GRAFICOS / "05_correlacion_caracteristicas.png", dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 05_correlacion_caracteristicas.png")
plt.close()

# 6. RESUMEN FINAL
print("\n" + "="*70)
print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
print("="*70)
print(f"\n📁 Gráficos guardados en: {CARPETA_GRAFICOS}")
print(f"   - 01_distribucion_actividades.png")
print(f"   - 02_distribucion_voluntarios.png")
print(f"   - 03_estadisticas_caracteristicas.png")
print(f"   - 04_boxplot_caracteristicas.png")
print(f"   - 05_correlacion_caracteristicas.png")
print(f"\n📄 Reporte de limpieza: {ruta_reporte_limpieza}")

print(f"\n📊 Resumen:")
print(f"   • Dataset: {len(datos.entrenamiento) + len(datos.prueba)} muestras totales")
print(f"   • Características: {datos.entrenamiento.shape[1]}")
print(f"   • Actividades: {len(datos.nombres_actividades)}")
print(f"   • Voluntarios: {len(np.unique(datos.voluntarios_entrenamiento))}")

print(f"\n🎯 Próximo paso: python scripts/02_entrenamiento.py")
print("="*70 + "\n")
