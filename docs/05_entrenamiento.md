# 5. Entrenamiento: schedule, sampler, driver y precisión

## 5.1 Schedule (`Hydra/schedule/`)

### Estructura anidada

`config0.schedule.learning_rate` contiene tres listas paralelas con la misma forma
`[eon][era][periodo]`:

| Nivel | Significado | Efecto |
|---|---|---|
| **eon** | Un stage de Hydra (`stage<k>` de config2) | Al terminar cada eon (salvo el último) la arquitectura evoluciona, se crea un `MCState` nuevo, se trasplantan pesos y el keeper abre una etapa nueva |
| **era** | Agrupación | Sin efecto propio |
| **periodo** | Una llamada `vmc.run(n_iter=épocas)` | Cada periodo crea **un optimizador y un driver nuevos**: el estado del optimizador (momentos de Adam), el momentum SPRING y el contador de pasos se reinician. Las cadenas de Markov no se reinician entre periodos, pero sí entre stages |

Por periodo:

* `epochs_struct[e][r][p]`: número de pasos.
* `modes_struct[e][r][p]`: lista de códigos que se entrenan (`["A"]` = toda la red).
* `lr_struct[e][r][p]`: una instrucción de learning rate por código.

`rescale` multiplica todos los learning rates (no el diag_shift). `total_epochs` es la suma
de todos los periodos; `stage_epochs[k]`, la de cada eon.

### Funciones de learning rate y diag_shift

Se escriben como cadenas `nombre(args)`; con `E` = épocas del tramo y `x = 0…E−1`:

| Sintaxis | Fórmula | Con `S` (escalera de S niveles) |
|---|---|---|
| número | constante | — |
| `lin(y0,y1[,S])` | y0 + (y1−y0)·x/E | k = ⌊xS/E⌋; y0 + (y1−y0)·k/S |
| `exp(y0,y1[,S])` | y0·(y1/y0)^(x/E) (y0, y1 ≠ 0, mismo signo) | y0·(y1/y0)^(k/S) |
| `sig(y0,y1,x_d[,S])` | y0 + (y1−y0)/(1+exp(−(x−x_d)/(E/40))): sigmoide centrada en la época `x_d`, anchura E/40 | (ver capítulo 11) |

No hay función `const`: para un diag_shift constante use `lin(a,a)`. Ni `lin` ni `exp`
alcanzan exactamente y1 (el último valor es el de x = E−1).

### Modos: qué parámetros se entrenan

Los códigos son los de Hydra (§4.3). `["A"]` debe ir solo y entrena todos los submódulos de
primer nivel: con un lr, compartido; con n lr, uno por submódulo en orden de código. Una
lista explícita (`["0","01"]`) requiere un lr por código. Los parámetros no seleccionados se
congelan (`optax.set_to_zero`). **Nota:** el SR calcula la dirección natural con todos los
parámetros y la máscara se aplica después; entrenar un subconjunto no equivale a hacer SR
restringido a ese subconjunto.

### diag_shift

* **Una cadena**: una curva sobre todas las épocas de la simulación (no se reinicia al
  cambiar de stage).
* **Una lista** con una cadena por eon: cada stage recorre su propia curva (útil en cadenas
  de sectores, donde cada stage vuelve a empezar). Una longitud distinta del nº de eons es un
  error.

Un número suelto no se acepta.

### Etiqueta del schedule

La carpeta `ST_…` se forma con las épocas y los modos de cada periodo:
`[[[700]],[[300]]]` con `["A"]` → `ST_700-A_300-A`. Los learning rates y el diag_shift **no**
aparecen: simulaciones que sólo difieren en ellos comparten carpeta y se distinguen por el
`UUID_…`.

### Ejemplo: dos stages, lr constante y luego decaimiento exponencial

```json
"schedule": {
  "learning_rate": {
    "epochs_struct": [ [[700]],      [[300]] ],
    "modes_struct":  [ [[["A"]]],    [[["A"]]] ],
    "lr_struct":     [ [[[0.01]]],   [[["exp(0.01,0.001)"]]] ],
    "rescale": 1.0
  },
  "diag_shift": ["lin(1e-3,1e-3)", "exp(1e-3,1e-4)"]
}
```

Resultado: 1000 épocas, etiqueta `ST_700-A_300-A`; en la época 700 se pasa de `stage0` a
`stage1` (config2 debe definir ambos). El keeper guarda un mejor estado por modo y stage.

Variante con dos periodos dentro de un stage y entrenamiento por bloques:
`epochs [[200, 500]]`, `modes [[["A"], ["0","1"]]]`,
`lrs [[[0.01], [0.01, "exp(0.01,0.001)"]]]`.

### Receta de referencia (J1-J2, redes con coset + Z2)

| Uso | Épocas | lr | diag_shift |
|---|---|---|---|
| GS 6x6 (local) | 700 + 300 | 0.007 → `exp(0.007,0.0005)` | `sig(0.01,0.0001,400)` |
| GS 8x8 (cluster) | 2000 + 1000 | 0.007 → `exp(0.007,0.0005)` | `sig(0.01,0.0001,1200)` |
| Stage de una cadena de sectores | 1400 + 600 | 0.007 → `exp(0.007,0.0005)` | sigmoide por stage (lista) |
| Cribado rápido | 200 + 100 | 0.007 → `exp(0.007,0.0005)` | `sig(0.01,0.001,120)` |

Con coset ningún run ha colapsado con un suelo de diag_shift de 1e-4; sin coset (Z2 sólo)
se observaron colapsos con suelo 1e-4 y conviene ≥ 1e-3.

## 5.2 Sampler (`Hydra/sampler/`)

`selection`: `metropolis` (`netket.MetropolisSampler`) o `nkARSampler` (muestreo directo
para redes autorregresivas). No hay *burn-in* (`n_discard_per_chain = 0`); el barrido entre
muestras es de N propuestas.

| Regla | Propuesta | Conserva Sz |
|---|---|---|
| `Exchange` | Intercambia el espín de un sitio con el de un vecino de una capa elegida al azar (≈64 % primeros, 29 % segundos, 6 % terceros vecinos) | Sí |
| `ExchangeJ1J2` | Intercambio NN (prob. J1/(J1+J2)) o NNN | Sí |
| `LocalRule` | Voltea un espín | No |
| `InvertMagnetization` | Voltea todos los espines de una cadena (movimiento global raro, mezclar con `prules`) | No (M → −M) |
| `LocalRule_Z2` | Voltea un espín salvo el sitio 0 | No |

`initial_state`: `hilbert` (aleatorio en el sector), `M0` (permutación de ±1 alternados,
M = 0), `Mmax` (todo arriba o abajo), `null`. Con `sz_total: 0` use `Exchange` y `hilbert`.

## 5.3 Drivers VMC (`Hydra/VMC/`)

### `vmc_sr_netket` (recomendado): `netket.driver.VMC_SR`

| Clave | Significado |
|---|---|
| `mode` | `"complex"` para log ψ complejo (estándar); `"real"` para funciones reales con signo |
| `linear_solver` | `cholesky_with_fallback` (defecto; Cholesky y, si da NaN, `pinv_smooth`), `cholesky`, `pinv`, `pinv_smooth`, `LU`, `cg`, `gmres`, `bicgstab` |
| `chunk_size_bwd` | Filas del NTK/Jacobiano por pasada. **Controla la memoria**: CvT3 8x8 con 2048 muestras por GPU necesitó 33 GiB sin trocear; con 256 cabe |
| `momentum` | Momentum SPRING μ ∈ [0,1] (≈0.8 recomendado por NetKet); se reinicia en cada periodo |
| `use_ntk`, `on_the_fly`, `proj_reg` | Se pasan a NetKet si se dan |

Si no se fija `use_ntk`, NetKet elige **NTK (minSR)** cuando hay más parámetros que muestras
y **QGT** en caso contrario (lo imprime: «Automatic SR implementation choice»).

### `vmc_netket`: `netket.driver.VMC`

`SR: true` usa `nk.optimizer.SR` con el `solver` indicado (`cg`, …); `SR: false` es
gradiente simple.

## 5.4 Flujo de una simulación

```
para cada tamaño de config1:
    carpeta UUID, espacio de Hilbert, ModelFactory
    para cada punto del barrido:
        H, ED (N ≤ 20), etiquetas
        Hydra(stage0) → MCState → (carga opcional de pesos)
        Schedule, callbacks (keeper obligatorio)
        para cada periodo:
            optimizador + driver nuevos → vmc.run
            si termina un eon: arch_evol → MCState nuevo → trasplante → keeper.new_stage
        keeper.finalize → restaurar el mejor estado (modo principal, último stage)
        gráficas, observables, JSON de resultados
```

Si la simulación cae (OOM, NaN, Ctrl-C o `scancel`), el keeper escribe antes de salir los
mejores estados pendientes y la traza; el JSON de resultados no se escribe, pero la carpeta
puede usarse como origen de `load_from` (se carga desde `_best.orbax`).

## 5.5 Precisión numérica

### `network_dtype`

* `"float64"` (defecto): todo en doble precisión.
* `"float32"`: **precisión mixta**. ViT2D, CvT3, la toolbox y OutputHead/DeepOutputHead
  calculan en float32; los parámetros, el Jacobiano, la resolución del SR, el optimizador,
  la proyección de momento y las energías siguen en float64. La inicialización es idéntica
  a la de float64 (mismos números para la misma semilla). Las etiquetas de los ficheros
  llevan `_f32`.

Rendimiento medido (RTX 3080, 6x6):

| Red | float64 | float32 mixto | Factor |
|---|---|---|---|
| ViT2D, 100 épocas | 7.5 s/it | 1.9 s/it | 3.9× (misma energía: −71.70 vs −71.73) |
| CvT3, 512 muestras | 17 s/it | 3.8 s/it | 4.5× (trayectoria igual a la de float64 época a época) |

En GPUs sin float64 rápido (GeForce, **L40S**) la ganancia es grande; en A100 (CESGA) el
float64 es rápido y la ganancia menor. Un float32 *puro* (también SR en float32) se volvió
inestable con diag_shift < 1e-3: por eso el SR siempre va en float64.

### `matmul_precision`

En GPUs Ampere y posteriores XLA usa por defecto **TF32** para los productos de matrices en
float32 (10 bits de mantisa). Para NQS no es suficiente: en log ψ produce errores de hasta 6
(inicialización) y 0.1–0.2 en redes entrenadas, un 3 % de error en el Jacobiano, y una CvT3
quedó atascada en −46 en lugar de −72. Por eso el valor por defecto es `"highest"` (float32
real, errores ~1e-4). `"default"` (TF32) y `"bfloat16"` existen sólo para pruebas. La
precisión de float64 no se ve afectada por este ajuste.

> **Lección aprendida (nov. 2026).** Los trabajos `_f32` lanzados antes de estas dos
> correcciones (TF32 por defecto e inicialización en float32) no son fiables y deben
> repetirse.

## 5.6 Memoria y tiempos de referencia

| Caso | Ajustes que funcionan | Tiempo |
|---|---|---|
| 6x6 ViT, RTX 3080 10 GB | 1024 muestras, `chunk_vstate 512` | 11 s/it (f64) |
| 6x6 CvT3, RTX 3080 | 1024 muestras, `chunk_vstate 512`, `chunk_bwd 64` | 24.5 s/it (f64) |
| 8x8 ViT, garmonegro 2×L40S | 4096 muestras | 16.9 s/it (f64, coset + Z2) |
| 8x8 CvT3, 2 GPUs | `chunk_bwd 256` obligatorio | ~2.2× la ViT |
| 10x10 en 10 GB | `chunk_vstate 1024` da OOM; 64 funciona | — |

`n_samples` debe ser divisible por `chunk_vstate`.
