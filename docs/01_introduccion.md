# 1. Introducción

**NNs** es un entorno de simulación de **estados cuánticos neuronales** (*Neural Quantum
States*, NQS) mediante **Monte Carlo variacional** (VMC). Calcula estados fundamentales y
estados excitados por sectores de simetría de modelos de espines en redes bidimensionales
(J1-J2, Shastry–Sutherland, Ising, interacciones de largo alcance, Kitaev-Γ en la red
triangular), comparando con diagonalización exacta cuando el tamaño lo permite.

Está construido sobre **NetKet 3.22** (muestreo Metropolis, reconfiguración estocástica,
operadores), **JAX** (GPU, diferenciación automática) y **Flax** (redes neuronales).

## Qué aporta sobre NetKet

| Componente | Función |
|---|---|
| **Configuración declarativa** | Cuatro JSON controlan toda la simulación: no hace falta tocar código para cambiar modelo, red, schedule o simetrías |
| **Hydra** (constructor de redes) | Ensambla redes a partir de plantillas (transformers ViT2D y CvT3, CNN, cabezas…), con proyección exacta sobre momentos (`Sequivariant` + coset) y paridad Z2, y permite **cambiar la arquitectura o el sector a mitad de simulación** trasplantando los pesos |
| **Schedule por bloques** | Learning rate y diag_shift por tramos, entrenamiento de submódulos concretos, stages |
| **BestIterKeeper** | Conserva en disco los mejores estados según varios criterios (energía, V-score o una puntuación conjunta) |
| **ED por sectores** | Diagonalización exacta sin matrices densas en cada sector de momento y Z2, para validar |
| **Precisión mixta** | Red en float32 con SR en float64: 4× más rápido en GPUs sin float64 rápido |
| **Resultados reproducibles** | Carpetas y nombres de fichero deterministas, JSON con toda la configuración, trazas en vivo |
| **Herramientas de campaña** | Generador de configuraciones y scripts de análisis para clusters (Slurm) |

## Mapa del manual

| Capítulo | Contenido |
|---|---|
| [2. Primeros pasos](02_primeros_pasos.md) | Instalación, primera simulación, qué se obtiene |
| [3. Configuración](03_configuracion.md) | Los cuatro JSON, clave por clave |
| [4. Redes (Hydra)](04_redes.md) | Plantillas, módulos, simetrías, stages, carga de pesos |
| [5. Entrenamiento](05_entrenamiento.md) | Schedule, sampler, drivers, precisión, memoria |
| [6. Callbacks](06_callbacks.md) | Mejores estados, checkpoints, diagnóstico |
| [7. Modelos y ED](07_modelos_y_ED.md) | Hamiltonianos, redes, diagonalización exacta |
| [8. Resultados y análisis](08_resultados_y_analisis.md) | Ficheros, JSON de resultados, observables, scripts |
| [9. Clusters y campañas](09_cluster_y_campanas.md) | `pile`, generador de campañas, estrategia |
| [10. Desarrollo y tests](10_desarrollo_y_tests.md) | Entorno, estructura, tests, convenciones |
| [11. Problemas conocidos](11_problemas_conocidos.md) | Errores frecuentes y limitaciones |

## Glosario

| Término | Significado |
|---|---|
| **Stage / eon** | Tramo de la simulación con una arquitectura y un sector fijos |
| **Periodo** | Tramo del schedule con un learning rate y unos módulos entrenados; cada uno crea un driver nuevo |
| **Código** | Identificador de un submódulo en el árbol de parámetros (`"0"`, `"01"`, `"A"` = todo) |
| **Coset** | Suma sobre los desplazamientos dentro del stride que convierte una red equivariante con stride en un autoestado exacto de momento |
| **Irrep** | Sector de simetría `(n_y, n_x[, z2])`: momento k = 2πn/L y paridad Z2 (0 par, 1 impar) |
| **V-score** | V = N·Var(E)/E²: medida intensiva de la calidad del estado |
| **minSR / NTK** | SR resuelto en el espacio de muestras (cuando hay más parámetros que muestras) |
| **Unidades de Pauli** | Hamiltoniano escrito con σ; para Heisenberg, E(Pauli) = 4·E(S) |
