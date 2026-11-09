# 6. Callbacks y mejores estados

Los callbacks se configuran en `config0.callback` (§3.1.1) y se construyen en
`Hydra/callback/__init__.py`. NetKet los ejecuta **antes** de actualizar los parámetros,
así que la energía registrada y el estado guardado corresponden exactamente a los mismos
parámetros. Ninguno vuelve a evaluar ⟨H⟩: leen la energía de `log_data`.

| Callback | Bandera | Para qué | Coste |
|---|---|---|---|
| **BestIterKeeper** | `keeper` (obligatorio) | Guarda los mejores estados según uno o varios criterios | Bajo |
| Checkpoint | `checkpoint` | Estado completo cada `do_each` pasos | Una escritura orbax por checkpoint; el disco crece |
| EnergyPlotter | `energy_plot` | Gráfica de energía en vivo | Muy alto (ver capítulo 11) |
| ModPhasePlotter | `modphase` | Módulo/fase de ψ en vivo | Muy alto, sólo sistemas pequeños (`to_array`) |
| SanityMonitor | `sanity` | Diagnóstico de gradientes, muestreo y fase | Alto |
| LeavesGradient | `leaves_gradient` | Estadísticas por hoja de parámetros (depuración) | Muy alto |

## 6.1 BestIterKeeper

### Criterios (modos)

Varios modos funcionan a la vez; cada uno conserva su propio mejor estado **por stage**.

| Modo | Un estado candidato es mejor si… |
|---|---|
| `best_energy` | Primero, los estados con V-score < `vscore_max` ganan a los que no la cumplen; después, menor energía. Si ninguno ha bajado aún de la cota, gana la menor energía |
| `best_vscore` | Menor V-score |
| `best` | Menor puntuación conjunta **S = E + \|E\|·V/c** |
| `always` | Siempre (el último estado; depuración) |

V = N·Var/E² es el V-score (intensivo e independiente del modelo).

**Puntuación conjunta.** Cerca de un autoestado, E − E₀ y la varianza miden el mismo error;
en términos relativos la varianza da ε_V = V/c. S suma ambas evidencias en unidades de
energía: acepta una pequeña subida de E si el V-score baja bastante (|E|·ΔV/c > ΔE) y una
bajada grande de E aunque V suba un poco. `c = 8` se calibró con simulaciones J1-J2 4x4 y
6x6 convergidas frente a ED (V/ε entre 5 y 16, mediana ≈ 8). `best_setup.z > 0` añade un
margen de ruido: el candidato debe mejorar en z·√((Var + Var_best)/N_s).

**Elegibilidad.** Dentro de cada stage sólo son elegibles los estados con
`paso_local > start_fraction × épocas_del_stage` (el primer estado siempre se toma, para que
cada modo tenga un mejor estado).

**Modo principal.** El primero de la lista `mode`. Al final de la simulación se restauran
los parámetros de su mejor estado en el **último stage**, y sobre ese estado se calculan
todos los resultados y observables.

### Almacenamiento en disco

En cada mejora sólo se toma una copia en RAM del host (`jax.device_get`); nada adicional
queda en la VRAM. Cada `save_every` pasos, en cada cambio de stage, al final y si la
simulación cae, los mejores pendientes se escriben con orbax (asíncrono):

```
<sim>_best.orbax/
    best/          parameters/  sampler_state/  metrics/      (un solo stage)
    best_energy/   ...
    best_vscore/   ...

<sim>_best.orbax/<modo>/stage<k>/{parameters, sampler_state, metrics}   (varios stages)
```

`metrics/metadata` es un JSON con `energy, variance, vscore, step, local_step, stage, mode,
score`. Cada carpeta se sobrescribe en sitio (escritura en carpeta temporal y renombrado:
una simulación que muere conserva el mejor anterior). Una carpeta `_best.orbax` antigua con
el mismo nombre se borra al empezar.

**Traza.** `<sim>_trace.txt` recoge cada paso:
`# step stage local_step energy variance vscore`. Permite seguir simulaciones largas en vivo
y no se pierde nada si el trabajo muere.

`orbax: false` mantiene los mejores sólo en RAM (sin traza; un fallo lo pierde todo).

### Parada

`survive_condition` detiene el **periodo** actual si el V-score baja de `baseline` o la
energía deja de ser finita (mensaje en `results.mssg`).

### Uso de los mejores estados

* Arrancar otra simulación: `load_from` admite la carpeta `_best.orbax` (o un nivel
  inferior; por defecto modo `best`, último stage) o directamente la carpeta del run (si no
  hay JSON de resultados se usa el `_best.orbax`).
* Analizar: `Hydra.saveNload.load_vstate(art, params_path=<…>/best_energy/parameters)`.
* `results.keeper` del JSON contiene `primary` y `bests[modo][stage]` con energía, V-score,
  paso, ruta y, si hay ED, el sector, `E_ED` y el error relativo de cada stage.

### API (para scripts)

| Método | Función |
|---|---|
| `update(step, log_data, driver)` | El callback |
| `flush(wait=True)` | Escribe los pendientes y la traza |
| `new_stage(stage)` | Cierra el stage actual y reinicia los criterios |
| `finalize()` | Cierra el último stage |
| `restore(vstate, mode=None)` | Carga en `vstate` el mejor del modo (último stage) |
| `best_path(mode, stage)` | Carpeta en disco de ese mejor |
| `summary()` | `{modo: [mejor de cada stage]}` |

## 6.2 Checkpoint

`checkpoint_setup = {"do_each": 500, "async": false}` escribe
`<sim>_checkpoint.orbax/<paso>/{parameters, sampler_state, metrics{energy, vscore}}`. Se
conservan todos los pasos (el disco crece). Se cargan con
`load_model.checkpoint = <paso>` y la ruta del `.orbax`. `plot_checkpoint.py` dibuja la
historia a partir de ellos.

## 6.3 Gráfica final (no es un callback)

Al terminar cada punto, `dump_callback` dibuja `Callback_<sim>.jpeg` con la energía (y las
líneas de ED), el error relativo (si hay ED), el V-score, las anotaciones del schedule y de
los stages, el mejor paso (punto dorado) y los checkpoints; y escribe
`…_energy.txt`, `…_energy_dev.txt` (error de la media), `…_vscore.txt` y, con ED,
`…_error.txt`.

## 6.4 SanityMonitor

`sanity_setup = {"do_each": 10, "mode": "simple"|"rich", "verbose": 0-4, "metrics":
{"gradients", "sampling", "sample_histogram", "phase"}}`. Diagnostica norma de gradientes,
aceptación, tiempo de correlación, ESS y estructura de la fase, con un semáforo
verde/amarillo/rojo y mensajes en castellano. Umbrales principales: aceptación fuera de
0.3–0.75, τ_corr > 10, eficiencia ESS < 0.1. La métrica `gradients` no funciona con los
árboles de parámetros actuales (capítulo 11).

## 6.5 LeavesGradient

Herramienta de depuración: cada `do_each` pasos imprime mín./máx. de Re log ψ, NaN/Inf y un
árbol con estadísticas por hoja (`metrics: "mean|std|max|min"`). Muy cara.
