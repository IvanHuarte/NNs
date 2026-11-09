# 11. Problemas conocidos y solución de errores

Estado a 9 de noviembre de 2026. Las entradas marcadas **[corregido]** ya están arregladas
en el código; el resto son limitaciones conocidas.

## 11.1 Errores frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `KeyError: 'baseline'` al construir callbacks | `keeper_setup` sin `baseline` | **[corregido]** ahora tiene valor por defecto 1e-30 |
| `RuntimeError` «keeper» | `callback.keeper: false` | El keeper es obligatorio |
| `RESOURCE_EXHAUSTED` en `srt_onthefly` | NTK demasiado grande | Bajar `chunk_size_bwd` (VMC) o `chunk_vstate`, o el nº de muestras; comprobar que no hay otro proceso en la GPU |
| `ptxas … could not be opened` | `TMPDIR` no escribible | `export TMPDIR=<carpeta propia>` |
| `KeyError` de un código en el schedule o en `load` | Código inexistente para esa arquitectura | El mensaje lista los códigos válidos; los de stages posteriores sólo se comprueban al llegar a ese stage |
| `KeyError` en `arch_evol` | El schedule tiene más eons que stages config2 | Añadir `stage<k>` o reducir `epochs_struct` |
| `FileNotFoundError` en `load_from` | La simulación origen no existe o no ha escrito nada | Usar una simulación terminada (o con `_best.orbax`) |
| Energía muy por encima de lo esperado con red float32 | TF32 o inicialización float32 | **[corregido]** `matmul_precision: "highest"` por defecto e inicialización idéntica a float64. Trabajos `_f32` anteriores al 9-nov-2026 deben repetirse |
| Error al cargar en CPU/1 GPU un estado de 2 GPUs | Sharding guardado | **[corregido]** se restaura como arrays numpy |
| ViT2D con tokens no cuadrados no da autoestados de momento | Reordenado de tokens | **[corregido]** |
| Colapso de la energía con diag_shift ~1e-4 (Z2 sin coset) | Inestabilidad del SR | Suelo de diag_shift ≥ 1e-3 sin coset |

## 11.2 Limitaciones y código a revisar

**Configuración**

* `optimizer.modifications.clip_by_global_norm` debe ser `null` (cualquier valor da
  `KeyError`).
* `change_attr` y `symm_wrapper` en `stage0` los ignora Hydra (el ED sí aplica el
  `symm_wrapper`): no los use en `stage0`.
* Entradas de storage con sufijo (`MLP_2`) nunca se leen; `Jastrow_wrap` no se puede usar
  desde una plantilla.
* En una plantilla, los atributos de un flujo (`irrep`, `coset`…) deben ir **después** de los
  submódulos; sólo se usa la primera clave raíz.
* `mode: "zip"` sin listas no ejecuta ningún punto.
* Oxalate con `J, K, Gamma` exige `"spherical": false`.
* `schedule.sampler` no se lee.
* Con `load_model.checkpoint ≠ 0` y un JSON de resultados la carga falla: use la ruta del
  `*_checkpoint.orbax`.
* El ED sólo detecta el sector si `Sequivariant` es la raíz de la plantilla.

**Entrenamiento**

* `sig(y0,y1,x_d,S)` con escalera compara el índice de nivel con `x_d` (en épocas): no
  decae. Use `sig` sin el cuarto argumento.
* Cuando el keeper detiene un periodo (V-score < baseline o NaN), los periodos siguientes y
  los cambios de stage continúan.
* `scancel`: la señal SIGTERM se convierte en una salida ordenada que guarda los mejores
  estados **[corregido]**, pero si un paso de XLA dura más que el margen de Slurm (30 s por
  defecto) el proceso puede ser terminado antes.

**Callbacks** **[corregido 9-nov-2026]**: Checkpoint asíncrono (la última escritura se
perdía), EnergyPlotter/ModPhasePlotter (no dibujaban en vivo y fallaban al guardar o sin
ED), SanityMonitor (métricas de gradientes sólo para SplitTraining, displays rotos) y
LeavesGradient se han reescrito (capítulo 6).

**Redes**

* Envoltorio `Traslation` (y `CarreteSign`): no funcional; use `Sequivariant` + `coset`.
* `VViT`, `EDPPh`: probablemente rotos bajo `NNWrapper`. `DeepOutputHead` no aplica sus
  activaciones. `MLP.only_phase: true` siempre falla. `Transversal` con una `operation`
  desconocida actúa como identidad.
* `CvT3`: `strides_qkv` y `n_mlp_layers` no se usan; el `kernel` sólo afecta al embedding.
* Sólo ViT2D, CvT3 y sus cabezas siguen `network_dtype`; el resto se queda en float64.

**Observables y scripts**

* `ms` es un parámetro de rayas, no de Néel, en redes de anchura par.
* La entropía de Rényi usa los primeros N/2+1 sitios.
* `load_vstate` construye un espacio de Hilbert sin restricción de Sz.
* `calc_SSF_NN.py`, `plot_SSF_NN.py`, `calc_observables_from_folder.py`,
  `batch_J1J2plotter.py`, `plot_J1J2simulations.py`, `plot_LRsweep.py`: usan claves del
  formato antiguo (`E_ED`, `error`, `fidelity`, `x_ED`, `coupling_model`) y fallan con los
  resultados actuales. `calc_observables_from_folder.py` además sobrescribe los JSON.
* `plot_phase.py`, `plot_SSF_NN.py`: `sys.exit(1, msg)` produce `TypeError`.
* `dynamic_plots.plot_modphase` sin ED falla; use `single_modphase_plot`.
* Los vectores de ED se reescriben siempre (la comprobación de existencia no añade `.txt`).
