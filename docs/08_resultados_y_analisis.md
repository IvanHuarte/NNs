# 8. Resultados y análisis

## 8.1 Estructura de carpetas

```
<write_folder_sim>/<Modelo>/<selección>_<plantilla del último stage>/Size_<L0>x<L1>/<ST_…>/
    UUID_<uuid>.json                 leyenda del barrido (se reescribe en cada punto)
    UUID_<uuid>/                     todos los puntos del barrido de este tamaño
        config_files/                los 4 JSON (con sizes = [tamaño])
        <ficheros de cada punto>
```

Ejemplo real:
`Simulations_claude/runs/6x6cvt/T_f64/J1J2Square/CvT3OH000f64_Sequivariant_CvT3_OutputHead_irrep_phase_mode_coset/Size_6x6/ST_60-A/UUID_6bf2e621/`

## 8.2 Etiquetas de los ficheros

| Etiqueta | Formato |
|---|---|
| `sim` | `<Modelo>_<selección>_<arq>_simulation_<L0>x<L1><parámetros>_date_<AAAAMMDDTHHMMSS>_UUID_<uuid>` |
| `ED` | `<Modelo>_<selección>_xED_<L0>x<L1><parámetros>_date_…_UUID_…` (sin arquitectura) |
| `json` | `<Modelo>_<selección>_<arq>_results_…` |

`<arq>` lista las clases de flujo y módulos de cada stage, sin atributos
(`Sequivariant_CvT3-OutputHead`; stages distintos unidos por `+`), más `_f32` si la red va en
precisión mixta. `<parámetros>` depende del modelo; los scripts de análisis lo parsean, así
que su formato es estable:

| Modelo | Fragmento |
|---|---|
| J1J2Square | `_J1J2_<J1>_<J2>_XYZ_<hx>_<hy>_<hz>` |
| SSMSquare | `_JJp_<J>_<Jp>_XYZ_…` |
| LRChain / LRSquare | `_J_<J>_alpha_<α>_XZ_<f0>_<f1>` |
| Oxalate | `_strength_<a>_theta_<θ>_phi_<φ>` |
| otros | `_<clave>_<valor>` por parámetro |

Ejemplo:
`J1J2Square_CvT3OH000f64_Sequivariant_CvT3-OutputHead_simulation_6x6_J1J2_1.0_0.5_XYZ_0.0_0.0_0.0_date_20261109T185737_UUID_6bf2e621`

## 8.3 Ficheros de cada punto

| Fichero | Contenido |
|---|---|
| `<json>.json` | Artefacto principal (§8.4) |
| `<sim>_vstate_parameters.orbax` | Parámetros del mejor estado (modo principal, último stage) |
| `<sim>_vstate_sampler_state.orbax` | Estado del sampler al final del entrenamiento |
| `<sim>_best.orbax/` | Mejores estados de cada modo (y stage) — capítulo 6 |
| `<sim>_trace.txt` | Energía, varianza y V-score de cada paso |
| `<sim>_checkpoint.orbax/<paso>/` | Checkpoints (si están activados) |
| `Callback_<sim>.jpeg` + `_energy.txt`, `_energy_dev.txt`, `_vscore.txt`, `_error.txt` | Gráfica e historia del entrenamiento |
| `<sim>_modphase_vstate.txt` | Módulo y fase (2 filas) |
| `<ED>.txt`, `<ED>irrep_[…].txt`, `<ED>_modphase_*.txt` | Vectores y módulo/fase del ED |

## 8.4 El JSON de resultados

Claves de nivel superior: `CM`, `NN`, `SIM`, `results`, `_artifacts`, `labels`, `metadata`.

| Clave | Contenido |
|---|---|
| `CM` | Tamaño, modelo, parámetros, `kwargs_lattice`, `S_operators` |
| `NN` | `name` = `<selección>_stage<k>` del mejor estado; `setup` = árbol completo de módulos (necesario para recargar la red) |
| `SIM` | Copia de config0 más `schedule.total_epochs`, `sampler.nsamples`, `VMC` |
| `_artifacts` | Rutas absolutas: `parameters`, `sampler_state`, `best`, `checkpoint`, `callback{…}`, `x_ED_global`, `x_ED_irrep`, `modphase{…}` |
| `labels` | Las etiquetas de §8.2 |
| `metadata` | uuid, fecha, versiones (python, numpy, scipy, netket, jax, flax), dispositivo |

`results`:

| Clave | Significado |
|---|---|
| `best_step` | Paso global del mejor estado |
| `E_best`, `E_best_per_site`, `vscore` | Energía MC, por sitio, y V-score en ese paso |
| `E_gr_global`, `E_gr_irrep`, `error_global`, `error_irrep`, `delta_irrep`, `irrep_info` | Comparación con ED (`null` si N > 20) |
| `fidelity_global`, `fidelity_irrep`, `fidelity_per_site_*` | \|⟨ψ_red\|ψ_ED⟩\| y su raíz N-ésima |
| `modphase` | Estadísticas de módulo y fase (red y ED) |
| `S_renyi` | Entropía de Rényi-2 (sólo sin `sz_total`) |
| `m`, `ms`, `m2`, `ms2` (y `_ED`) | Magnetizaciones (ver nota) |
| `energy_units` | `"Pauli"` o `"S"` |
| `time_exe`, `seed`, `key`, `mssg` | Tiempo de entrenamiento, semilla, mensaje de salida |
| `schedule_flat` | Schedule aplanado (épocas, modos, lr, límites de eon) |
| `keeper` | `primary` y `bests[modo][stage]` con energía, V-score, paso, ruta y, con ED, sector, `E_ED` y error |

Nota: `ms` usa el signo (−1)^i del índice de sitio; en redes de anchura par con `default_2`
corresponde a un patrón de **rayas**, no de Néel.

## 8.5 Observables sobre una simulación guardada

```python
import json, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from Hydra.saveNload import load_vstate
from Hydra.observables import calc_all_observables_vs
from Hydra.correlations import correlations_vstate

art = json.load(open(".../<json_label>.json"))
vs = load_vstate(art)                                  # mejor estado (modo principal)
# otro mejor estado:  load_vstate(art, params_path=art["_artifacts"]["best"] + "/best_energy/parameters")
vs.chunk_size = 512                                    # trocear en redes grandes
S2, m, ms, m2, ms2 = calc_all_observables_vs(vs)       # unidades de Pauli
C = correlations_vstate(vs)                            # <σ_i·σ_j>, i <= j
```

`load_vstate` restablece la precisión de la red y reconstruye modelo, sampler y estado.
Construye un espacio de Hilbert **sin** restricción de Sz: el muestreo con reglas
`Exchange` se queda en el sector entrenado, pero `to_array()`, módulo/fase y fidelidades
recorrerían el espacio completo y no son comparables con los valores del JSON.

Otras utilidades (`Hydra/observables.py`): `distance_with_neel`, `distance_with_stripped`,
`print_max_contributors`, `full_basis_state` (embebe un vector del sector Sz en la base
completa). Gráficas (`Hydra/dynamic_plots.py`): `single_modphase_plot`,
`plot_M_sector_hist`, `plot_cum_contributors`, `plot_weighted_phase_hist`.

## 8.6 Scripts de la raíz

| Script | Uso | Estado |
|---|---|---|
| `VMC_simulation.py` | Simulación (capítulo 2) | Mantenido |
| `ED_inspection.py` | Espectro ED de los modelos de config1 | Mantenido |
| `plot_phase.py -a <json> [-w carpeta]` | Módulo y fase de la red y del ED | Funciona con el formato actual |
| `plot_checkpoint.py -a <json o .orbax> [-m 1]` | Historia a partir de checkpoints | Parcialmente actualizado |
| `batch_J1J2plotter.py -d <carpeta> [-o salida -j N]` | Gráficas de barridos J2/J1 en paralelo | Usa claves antiguas (`E_ED`, `error`) |
| `plot_J1J2simulations.py`, `plot_LRsweep.py`, `plot_SSF_NN.py`, `calc_SSF_NN.py`, `calc_observables_from_folder.py` | Análisis de formatos anteriores | Obsoletos (capítulo 11) |
| `experiment.py`, `experiment.ipynb`, `tests_oxalate.py`, `quick_example.py`, `main.py` | Pruebas y borradores | No mantenidos |

## 8.7 Análisis de campañas

`Simulations_claude/analyze_cluster2.py [RAÍZ]` y `analyze_local_cvt.py` leen las trazas del
keeper (también de trabajos en marcha o muertos), los JSON y los mejores estados en disco,
y tabulan energía a 1/100/300/500/2000 épocas, épocas hasta converger a 0.1 del valor final,
mejor energía, comparaciones entre sectores y parejas float64/float32 (capítulo 9).

## 8.8 Post-procesado: `aplot`

`autoplots/aplot.py` dibuja y tabula cualquier conjunto de experimentos a partir de un
**índice**: una fila por JSON de resultados y una columna por atributo. Alias cómodo:
`alias aplot=~/Escritorio/Ivan/NNs/autoplots/aplot.py`.

```bash
aplot y=E_site_spin x=J2 hue=L --where "net='CvT3' and dtype=='float32' and not vscore>0.1"
aplot y=E_site_spin x=J2 hue=L --where "irrep=='000'" --app --where "irrep==001"   # 2 selecciones, mismos ejes
aplot y=gap x=irrep hue=L --where "net='ViT2D'" --for dtype                         # una figura por dtype
aplot y=E_site_spin x="1/L**3" fit=lin --where "irrep=='000'"                        # tamaño finito
aplot y=E_site_spin x=var fit=lin hue=L                                              # extrapolación en varianza
aplot trace y=E_site_spin smooth=50 sched=1 --where "job~='c8_CvT3'"                 # trazas + lr y diag_shift
aplot table sort=E50 --where "L==10" fmt=md                                          # ranking (md, tex, csv)
```

- **Índice** (`autoplots/index.py`, en caché en `autoplots/.cache/`; `--refresh` lo rehace).
  Lee los cuatro esquemas de JSON. Columnas:
  - `job`, `host`, `L`, `N`, `size`, `system`;
  - cada parámetro de `CM.params` (también de modelos futuros) y cada escalar de `results`;
  - `name`, `net`, `head`, `irrep` (código `441`), `kx`, `ky`, `z2`, `dtype`, `seed`, `nsamples`, `epochs`;
  - `E`, `E_spin`, `E_site`, `E_site_spin`, `var`;
  - de la traza: `E50` (mejor media de 50 épocas del último stage, sin las energías divergidas), `vscore50`, `nan_epoch`;
  - `E_ref`, `err_rel` (ED del sector o `autoplots/refs.json`), `time_h`, `s_it`, `status` (`ok`, `diverged`, `nan`).

  Cualquier otro nombre se busca en los JSON (`find_value`; vale una ruta `` `SIM/optimizer/optimizer` ``).
  `gap` = `E_spin` menos la menor `E_spin` del mismo sistema dentro de la selección.
- **`--where`**: consulta de pandas con `and`, `or`, `not`, `in [...]`, más tres atajos: `a=b` → `a==b`,
  `a~='regex'` (contiene) e `irrep==001` (el número se compara como texto).
- **`--app`** abre otra selección en los mismos ejes; hereda las claves de la anterior salvo
  `--where` y `--label`. El color sigue a `hue` y el estilo de línea a la selección.
  **`--for COL[,COL]`** genera una figura por valor presente.
- Varias runs en el mismo punto: `agg=best` (la de menor `best=E`, por defecto), `mean`, `median` o `all`.
  Nunca se borra nada.
- Cada figura se guarda (`-o Figures`, `--fmt png|pdf|svg`) junto a un `.csv` con los valores
  dibujados. Ambos llevan el comando y la versión de git.
- Claves de cada tipo: docstrings de `autoplots/plots/{xy,trace,table}.py`.
  Desde Python: `from autoplots.aplot import main; main("...")`.
  Hay un notebook plantilla en `autoplots/explore.ipynb`.
- `aplot` no usa la GPU (`JAX_PLATFORMS=cpu`).

**Re-evaluación de los mejores estados.** `E_best` es el mínimo de muchas estimaciones
ruidosas: está sesgado hacia abajo. Para el valor publicable:

```bash
autoplots/reeval.py -d Simulations_claude_cesga --where "L==8 and status=='ok'" --samples 65536
```

El script carga `<sim>_best.orbax/<modo>` y escribe `<json>_reeval.json` con E ± error, varianza,
V-score, τ y R̂. El índice lo expone como `E_re`, `E_re_err`, `vscore_re`…, por ejemplo
`aplot y=E_re x=L yerr=E_re_err`. En el clúster usa la GPU.

`autoplots/autoplot.py` (`config_plots.json`, `energyVSirrep`) sigue funcionando: energías por
sitio desde `E_best_per_site`, grupos ordenados por valor y nombres de parámetro libres.
