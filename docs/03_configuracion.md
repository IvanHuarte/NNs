# 3. Configuración: los cuatro JSON

Toda simulación se controla con cuatro ficheros JSON, que se pasan **en este orden**:

| Orden | Fichero | Contenido |
|---|---|---|
| 1 | `config0.json` | Simulación: carpeta de salida, sector de Sz, callbacks, sampler, optimizador, driver VMC, schedule, precisión |
| 2 | `config1_CM.json` | Modelo físico (*coupling model*): red, tamaños, Hamiltoniano, barridos de parámetros |
| 3 | `config2_Hydra.json` | Red neuronal: plantilla de cada stage, carga de pesos |
| 4 | `config3_Hydra_NN.json` | Hiperparámetros de los módulos (*storage*) y envoltorios de simetría |

```bash
./VMC_simulation.py -c config0.json -c config1_CM.json -c config2_Hydra.json -c config3_Hydra_NN.json
```

Sin `-c` se usan los cuatro ficheros de la raíz del repositorio. Hay que pasar exactamente
cuatro. Los JSON no admiten comentarios.

## 3.0 El patrón `selection`

Varias entradas contienen una clave `"selection"` y varios sub-dicts hermanos: **sólo se
usa el hermano nombrado por `selection`**; los demás son un catálogo inactivo (por eso los
ficheros contienen muchas configuraciones guardadas que no afectan a la simulación).

| Entrada | Opciones válidas |
|---|---|
| config0 `sampler.selection` | `metropolis`, `nkARSampler` |
| config0 `VMC.selection` | `vmc_sr_netket`, `vmc_netket` |
| config1 `CM.selection` | `J1J2Square`, `SSMSquare`, `IsingSquare`, `LRSquare`, `LRChain`, `Oxalate` |
| config2 `selection` | El nombre de cualquier evolución definida en el fichero |

## 3.1 config0.json — simulación

| Clave | Tipo | Defecto | Significado |
|---|---|---|---|
| `write_folder_sim` | str | obligatorio | Carpeta raíz de resultados. **Debe terminar en `/`** |
| `sz_total` | int / null | obligatorio | Restricción de Sz total del espacio de Hilbert (`0` para el sector M=0; `null` sin restricción). Use `null` con Hamiltonianos que no conservan Sz (campos X/Y, Ising, LR, Oxalate) |
| `seed` | int / null | hora actual | Semilla base. De ella se derivan (con `fold_in` por stage) las semillas de parámetros y sampler |
| `network_dtype` | `"float64"` / `"float32"` | `"float64"` | Precisión de la red (§5.5) |
| `matmul_precision` | `"highest"`, `"default"`, `"tensorfloat32"`, `"bfloat16"` | `"highest"` | Precisión de los productos de matrices float32 (§5.5). No cambiar |
| `callback` | dict | obligatorio | §3.1.1 y capítulo 6 |
| `sampler` | dict | obligatorio | §3.1.2 y §5.2 |
| `optimizer` | dict | obligatorio | §3.1.3 |
| `VMC` | dict | obligatorio | §3.1.4 y §5.3 |
| `schedule` | dict | obligatorio | §3.1.5 y §5.1 |

### 3.1.1 `callback`

Las seis banderas `keeper`, `energy_plot`, `modphase`, `sanity`, `leaves_gradient` y
`checkpoint` deben estar presentes; cada `<nombre>_setup` sólo se lee si su bandera es
`true`. **`keeper` debe ser `true`** (los resultados se calculan sobre el mejor estado).

```json
"callback": {
  "keeper": true, "energy_plot": false, "modphase": false,
  "sanity": false, "leaves_gradient": false, "checkpoint": false,
  "keeper_setup": {
    "mode": ["best", "best_energy", "best_vscore"],
    "baseline": 1e-30,
    "orbax": true,
    "save_every": 50,
    "start_fraction": 0.1,
    "vscore_max": 0.05,
    "best_setup": {"c": 8.0, "z": 0.0}
  },
  "checkpoint_setup": {"do_each": 500, "async": true, "max_to_keep": null, "stage_end": true},
  "sanity_setup": {"do_each": 50, "mode": "simple", "verbose": 2,
                   "metrics": {"energy": true, "sampling": true, "optimization": true,
                               "wavefunction": true}}
}
```

| `keeper_setup` | Defecto | Significado |
|---|---|---|
| `mode` | obligatorio | Uno o varios de `best`, `best_energy`, `best_vscore`, `always`. **El primero es el modo principal** (el que se usa para los resultados) |
| `baseline` | `1e-30` | Se detiene el periodo si el V-score cae por debajo |
| `orbax` | `true` | Guardar los mejores estados en disco (`<sim>_best.orbax`) y la traza `<sim>_trace.txt` |
| `save_every` | `50` | Pasos entre escrituras |
| `start_fraction` | `0.1` | Fracción inicial de cada stage que no es elegible |
| `vscore_max` | `0.05` | Cota de V-score de `best_energy` (`null`: sin cota) |
| `best_setup` | `{"c": 8, "z": 0}` | Parámetros de la puntuación conjunta del modo `best` |
| `recover` | `{"enabled": true, "window": 50, "rel_gap": 5e-3, "var_ratio": 3}` | Al cambiar de stage, partir del mejor estado del stage sólo si su final es inestable (capítulo 6). `false` lo desactiva |

Detalles de cada modo y del resto de callbacks en el capítulo 6.

### 3.1.2 `sampler`

```json
"sampler": {
  "selection": "metropolis",
  "n_chains_per_rank": 1024, "n_samples_per_chain": 1,
  "chunk_sampler": null, "chunk_vstate": 512,
  "metropolis": {
    "sampler": {"rules": ["Exchange"], "prules": [1.0], "initial_state": "hilbert"},
    "n_chains_per_rank": 1024, "n_samples_per_chain": 1
  },
  "nkARSampler": {}
}
```

Las claves no-dict dentro del bloque seleccionado **sustituyen** a las generales del mismo
nombre (se avisa por pantalla). Número total de muestras por paso:

```
n_samples = nº de GPUs × n_chains_per_rank × n_samples_per_chain
```

| Clave | Significado |
|---|---|
| `n_chains_per_rank` | Cadenas de Markov por GPU |
| `n_samples_per_chain` | Muestras por cadena y paso |
| `chunk_sampler` | Cadenas evaluadas por pasada durante el muestreo (debe dividir `n_chains_per_rank`; `null` sin trocear) |
| `chunk_vstate` | Muestras por trozo en energías y gradientes; también el `chunk_size_bwd` por defecto del SR. `n_samples` debe ser divisible por él |
| `metropolis.sampler.rules` | Reglas de propuesta (§5.2) |
| `metropolis.sampler.prules` | Probabilidad de cada regla (se ignora con una sola regla) |
| `metropolis.sampler.initial_state` | `hilbert`, `M0`, `Mmax` o `null` |

### 3.1.3 `optimizer`

```json
"optimizer": {"optimizer": "sgd", "modifications": {"clip": null, "clip_by_global_norm": null}}
```

| Clave | Valores |
|---|---|
| `optimizer` | `"sgd"` (recomendado con SR) o `"adam"` |
| `modifications` | `clip`, `clip_by_block_rms`, `scale`, `scale_by_rms`, `scale_by_learning_rate`, `add_noise`, `trace`: `null`/`false` desactiva; un escalar se pasa como argumento; una lista como `*args`. `clip_by_global_norm` debe quedarse en `null` (ver capítulo 11) |

### 3.1.4 `VMC`

```json
"VMC": {
  "selection": "vmc_sr_netket",
  "vmc_sr_netket": {"mode": "complex", "linear_solver": "cholesky_with_fallback",
                    "chunk_size_bwd": 256, "momentum": null},
  "vmc_netket": {"SR": true, "solver": "cg"}
}
```

Opciones en §5.3.

### 3.1.5 `schedule`

```json
"schedule": {
  "learning_rate": {
    "epochs_struct": [[[1400, 600]]],
    "modes_struct":  [[[["A"], ["A"]]]],
    "lr_struct":     [[[[0.007], ["exp(0.007,0.0005)"]]]],
    "rescale": 1.0
  },
  "diag_shift": "sig(0.01,0.0001,800)"
}
```

Estructura `[eon][era][periodo]`; un eon es un stage de Hydra. Explicación completa,
funciones disponibles y ejemplos en §5.1. (`schedule.sampler` no se lee.)

## 3.2 config1_CM.json — modelo físico

```json
{
  "mode": "for",
  "sizes": [[6, 6]],
  "kwargs_lattice": {"bc": "periodic", "order": "default_2"},
  "S_operators": false,
  "CM": {
    "selection": "J1J2Square",
    "J1J2Square": {"J1": 1.0, "J2_list": [0.4, 0.5, 0.6], "fields": [0.0, 0.0, 0.0]}
  }
}
```

| Clave | Significado |
|---|---|
| `sizes` | Lista de tamaños `[L0, L1]`. Cada tamaño es una iteración externa con su propia carpeta `UUID_…`, su espacio de Hilbert y su ED |
| `mode` | Barrido de las claves `*_list`: `"for"` = producto cartesiano; `"zip"` = emparejado elemento a elemento (trunca a la lista más corta; sin listas no se ejecuta ningún punto) |
| `kwargs_lattice` | `bc`: `"open"` (defecto) o `"periodic"`; `order`: `default_1` (defecto), `default_2`, `snake_1`, `snake_2`. Use **`default_2` + `periodic`** con redes equivariantes y ED por sectores |
| `S_operators` | `false`: Hamiltoniano con matrices de Pauli σ (unidades por defecto); `true`: con S = σ/2 (campos ×1/2, acoplos ×1/4) |
| `CM.selection` | Modelo (capítulo 7) |
| `CM.<Modelo>` | Parámetros. Una clave `X_list` se barre (`J2_list` → `J2`); el resto son constantes |

Las claves de nivel superior que no son las anteriores (`Oxalate`, `J2range`, …) no se
leen: son almacén personal.

> **Unidades.** Para modelos de Heisenberg, H(Pauli) = 4·H(S). Las energías de los
> resultados están en las unidades del Hamiltoniano construido; el JSON lo indica en
> `results.energy_units`. Ejemplo: el estado fundamental 6x6 J1-J2 (J2 = 0.5) es −72.5487 en
> unidades de Pauli, es decir E/N = −0.5038 en unidades de S.

## 3.3 config2_Hydra.json — red y etapas

| Clave | Significado |
|---|---|
| `selection` | Evolución activa. Su nombre aparece en carpetas y ficheros. Convención: `<Módulo><Cabeza><n_y n_x z2>` + sufijo (`CvT3OH000`, `ViT2DOH441w`) |
| `load_model.parameters` / `.sampler` | Cargar pesos / estado del sampler desde `load_from` |
| `load_model.checkpoint` | Paso de un `*_checkpoint.orbax` (0 = estado final/mejor) |
| `load_model.load_from` | Lista de orígenes: JSON de resultados, carpeta, glob, `*_best.orbax` u `.orbax` |
| `<evolución>.stage<k>` | Un stage por eon del schedule, numerados consecutivamente desde `stage0` |

Claves de cada stage:

| Clave | stage0 | stage ≥ 1 | Significado |
|---|---|---|---|
| `template` | ✓ | ✓ | Arquitectura (§4.2) |
| `save_params` | ✓ | ✓ | Conservar los parámetros finales para etapas posteriores |
| `load` | obligatorio si `load_model` carga algo | opcional (por defecto trasplanta todo de la etapa anterior; `[]` = aleatorio) | Instrucciones de trasplante `[origen, código_viejo, código_nuevo]` |
| `change_attr` | ignorado | ✓ | Cambiar atributos de módulos |
| `symm_wrapper` | **no usar** (Hydra lo ignora pero el ED sí lo aplica) | ✓, acumulativo | Cambiar opciones de Z2/Traslation (p. ej. pasar a Z2 impar) |

Detalles en el capítulo 4.

## 3.4 config3_Hydra_NN.json — storage y simetrías

| Clave | Significado |
|---|---|
| `print_arch` | `true`: imprimir el árbol completo de parámetros |
| `symm_wrapper` | `Traslation` y `Z2` (ambas claves deben existir). Cada una con `"on"` y sus opciones. El primero activo es el más externo |
| `storage` | Hiperparámetros por nombre de clase de módulo |

## 3.5 Ejemplo completo mínimo

Estado fundamental de J1-J2 6x6 con CvT3 + coset + Z2 par, 1000 épocas, en precisión
mixta. config2:

```json
{
  "selection": "CvT3OH000",
  "load_model": {"parameters": false, "sampler": false, "checkpoint": 0, "load_from": []},
  "CvT3OH000": {
    "stage0": {"template": {"Sequivariant": {"CvT3": {}, "OutputHead": {}, "irrep": [0, 0],
                                             "phase_mode": "angle", "coset": [2, 2]}},
               "save_params": true}
  }
}
```

config3 (extracto): `symm_wrapper.Z2 = {"on": true, "trivial_Z2": true,
"symm_modulus": false}`, `storage.CvT3` con `kernel [[3,3],[3,3]]` y
`strides [[2,2],[1,1]]`, `storage.OutputHead = {"d_model": 60, "only_phase": false}`.

config0 (extracto): `sz_total: 0`, `network_dtype: "float32"`, sampler 1024 cadenas × 1,
`chunk_vstate: 512`, `VMC.vmc_sr_netket.chunk_size_bwd: 64`, schedule 700 + 300 épocas con
`lr 0.007 → exp(0.007,0.0005)` y `diag_shift: "sig(0.01,0.0001,400)"`.

La forma más cómoda de generar configuraciones coherentes es el generador
`Simulations_claude/gen.py` (capítulo 9), que parte de los cuatro ficheros de la raíz y
aplica sólo los cambios necesarios.
