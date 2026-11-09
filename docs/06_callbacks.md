# 6. Callbacks y mejores estados

Los callbacks se configuran en `config0.callback` (§3.1.1) y se construyen en
`Hydra/callback/__init__.py`. NetKet los ejecuta **antes** de actualizar los parámetros,
así que la energía registrada y el estado guardado corresponden exactamente a los mismos
parámetros. Ninguno vuelve a evaluar ⟨H⟩: leen la energía de `log_data`.

| Callback | Bandera | Para qué | Coste |
|---|---|---|---|
| **BestIterKeeper** | `keeper` (obligatorio) | Guarda los mejores estados según uno o varios criterios | Bajo |
| Checkpoint | `checkpoint` | Estado completo cada `do_each` pasos, escritura asíncrona | Una copia a RAM por checkpoint |
| SanityMonitor | `sanity` | Diagnóstico de salud de la simulación (cualquier arquitectura) | De casi nulo a un Jacobiano en una submuestra, según las métricas |
| LeavesGradient | `leaves_gradient` | Paso relativo por módulo/hoja, hojas atípicas y mapa de calor final | Una copia de parámetros + Jacobiano opcional en una submuestra |
| EnergyPlotter | `energy_plot` | Ventana en vivo de energía, V-score y error; PNG | Bajo (redibuja cada `refresh_each`) |
| ModPhasePlotter | `modphase` | Módulo y fase de ψ en vivo frente al ED; PNG | `to_array` en sistemas pequeños; muestras en grandes |

Todos los callbacks reciben de `VMC_simulation.train` tres avisos opcionales:
`new_stage(stage)` al cambiar de stage, `flush(wait=True)` si la simulación cae (OOM,
Ctrl-C, `scancel`) y `finalize()` al terminar.

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

```json
"checkpoint_setup": {"do_each": 500, "async": true, "max_to_keep": null,
                     "keep_every": null, "stage_end": true}
```

Escribe `<sim>_checkpoint.orbax/<paso global>/{parameters, sampler_state, metrics}` con
métricas `energy, energy_error, variance, vscore, stage, local_step`.

* **`async: true`** (por defecto): el callback sólo copia el estado a la RAM del host
  (milisegundos) y orbax escribe en segundo plano mientras sigue el entrenamiento. Hay a lo
  sumo una escritura en curso; al cambiar de stage, al terminar y si la simulación cae se
  espera a que acabe, así que el último checkpoint siempre queda completo. (Antes nadie
  esperaba al final y la última escritura asíncrona se perdía.)
* `max_to_keep`: conservar sólo los n más recientes; `keep_every`: además, los de pasos
  múltiplos de este número.
* `stage_end`: guardar también el último paso de cada stage (útil para reanudar cadenas).
* Carga: `load_model = {"checkpoint": <paso>, "load_from": ["…_checkpoint.orbax"]}`.

## 6.3 SanityMonitor

```json
"sanity_setup": {"do_each": 50, "mode": "rich", "verbose": 2,
                 "metrics": {"energy": true, "sampling": true, "optimization": true,
                             "wavefunction": true, "gradients": false,
                             "local_energy": false, "magnetization": false},
                 "module_depth": 1, "n_jacobian": 128, "log": true}
```

Cada `do_each` pasos calcula las métricas, las clasifica como `ok`, `info`, `warn` o
`error` y las muestra; con `log: true` añade cada comprobación a `<sim>_sanity.jsonl`. Las
métricas por módulo usan los mismos códigos que el schedule (`module_depth: 1` = submódulos
de primer nivel, p. ej. backbone y cabeza; 2 = sus hijos, p. ej. los StageBlock de CvT3).

| Métrica | Qué mide | Coste | Avisos |
|---|---|---|---|
| `energy` | E ± σ, V-score, Im E en unidades de σ, ΔE y cociente de V-score respecto a la comprobación anterior, error frente al ED | Nulo | Im E > 5σ; subida de E > 3σ (error > 10σ); V-score ×3 |
| `sampling` | Aceptación, R̂, τ_corr (de las estadísticas de NetKet), ESS/N, fracción de muestras distintas | Casi nulo | Aceptación < 0.15 (< 0.05: cadenas congeladas) o > 0.95; R̂ > 1.1; τ > 5; muestras repetidas en espacios grandes. R̂ y τ sólo se evalúan con cadenas de ≥ 32 muestras |
| `optimization` | Por módulo: \|θ\|, dirección SR \|dp\| y su reparto, **paso aplicado \|Δθ\|/\|θ\|** (incluye lr, momentum y máscaras), lr efectivo \|Δθ\|/\|dp\|, diag_shift, NaN | Casi nulo | Paso relativo > 0.05 (error > 0.5); un módulo que se mueve > 100× la mediana; NaN/Inf |
| `wavefunction` | σ y rango de log\|ψ\|² en las muestras, armónicos de la fase \|⟨e^{iqφ}⟩\| (q = 1…4) | Una pasada | log ψ no finito; σ[log\|ψ\|²] > 10 (pocas configuraciones dominan) |
| `gradients` | En una submuestra (`n_jacobian`): \|O_re\| (sensibilidad del módulo) y \|O_im\| (de la fase) por módulo, parámetros muertos, espectro del NTK frente a diag_shift (direcciones que SR resuelve, condicionamiento) | Jacobiano O(n_jacobian·P) | > 20 % de parámetros muertos; < 5 % del espectro por encima de diag_shift (SR ≈ gradiente lento); λ_max/diag_shift > 1e10 |
| `local_energy` | Colas de E_loc (fracción > 5σ, máximo, curtosis) | Un paso extra | > 1 % de valores a > 5σ (estimador ruidoso: nodos de ψ) |
| `magnetization` | M media, sectores visitados, histograma | Nulo | Informativo |

Los escalares cuya derivada es exactamente cero son constantes globales de log ψ
(normalización o fase global, p. ej. el sesgo de la capa final) y no cuentan como muertos.

**Displays.** `mode: "rich"`: un panel por categoría con tablas de colores; `"simple"`:
texto plano alineado, adecuado para los logs de Slurm. `verbose`: 0 nada; 1 sólo errores;
2 avisos y errores (por defecto); 3 todas las comprobaciones; 4 además los comentarios y las
tablas por módulo. Los nombres antiguos de config (`phase`, `sample_histogram`) siguen
funcionando.

## 6.4 LeavesGradient

```json
"leaves_gradient_setup": {"do_each": 50, "depth": 2, "jacobian": true, "n_jacobian": 64,
                          "outlier": 100, "show": "tree"}
```

* Cada `do_each` pasos imprime un árbol agregado a `depth` (0 = todas las hojas) con
  `nº | |θ| | |dp|/|θ| | |Δθ|/|θ| | |O_re| | |O_im| | muertos` y una **regla** que sitúa el
  paso relativo de cada fila respecto a la mediana (centro `│`, ±3 décadas): las capas que
  aprenden mucho más rápido o más lento que el resto se ven de un vistazo.
* Lista aparte las **hojas atípicas** (factor `outlier` sobre o bajo la mediana, congeladas,
  muertas o no finitas), sea cual sea `depth`. `show: "outliers"` sólo muestra esa lista;
  `"none"`, nada (sólo el mapa final).
* Al terminar guarda **`<sim>_leaves_heatmap.png`** (hojas × pasos, color = log10
  \|Δθ\|/\|θ\|, líneas blancas en los cambios de stage) y los datos en `<sim>_leaves.npz`:
  muestra qué capas aprenden en cada momento, cuáles se congelan o se disparan y el efecto
  de cada periodo del schedule.

## 6.5 EnergyPlotter y ModPhasePlotter

Ventanas en vivo cuando matplotlib tiene un backend gráfico (QtAgg con `DISPLAY`); además, o
en su lugar, escriben `<sim>_live_energy.png` y `<sim>_live_modphase.png` (cada
`savefig_each` pasos y al final). Sin pantalla ni ruta de salida se desactivan solos.
`interactive: true/false` fuerza el modo.

**EnergyPlotter** (`refresh_each` 10, `savefig_each` 0, `window` 50, `max_points` 4000):
energía de cada paso y media móvil (reiniciada en cada stage), mejor media del stage,
líneas de ED del sector y global, V-score y error relativo de la media (o σ de cada paso
sin ED), con los cambios de stage marcados. Sólo actualiza los datos de las curvas existentes.

**ModPhasePlotter** (`plot_each` 50, `max_states` 65536, `order` `"basis"`/`"ed"`,
`no_null_mod`, `logscale`, `bins` 200):

* *Modo exacto* (espacio de Hilbert ≤ `max_states`): ψ normalizada; con ED del mismo sector
  se elimina la fase global arbitraria (ψ·e^{−i arg⟨ψ_ED|ψ⟩}), de modo que las fases se
  comparan punto a punto, y se muestra la fidelidad. Paneles: \|ψ_i\| con las
  configuraciones muestreadas como marcas; fase por configuración (red y ED); histograma de
  la fase **ponderado por \|ψ\|²** (red y ED).
* *Modo muestras* (sistemas grandes): histogramas de log\|ψ\|² y de la fase sobre las
  muestras (que ya siguen \|ψ\|²).

## 6.6 Gráfica final (no es un callback)

Al terminar cada punto, `dump_callback` dibuja `Callback_<sim>.jpeg` con la energía (y las
líneas de ED), el error relativo (si hay ED), el V-score, las anotaciones del schedule y de
los stages, el mejor paso (punto dorado) y los checkpoints; y escribe
`…_energy.txt`, `…_energy_dev.txt` (error de la media), `…_vscore.txt` y, con ED,
`…_error.txt`.

