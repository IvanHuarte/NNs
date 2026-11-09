# 2. Primeros pasos

## 2.1 Preparar el entorno

```bash
cd ~/Escritorio/Ivan/NNs
uv sync                      # crea/actualiza .venv a partir de pyproject.toml y uv.lock
export TMPDIR=$HOME/tmp      # carpeta escribible para la compilación de XLA en GPU
JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests -q    # comprobación (145 tests)
```

## 2.2 Primera simulación: J1-J2 4x4

1. **config1_CM.json**: `"sizes": [[4, 4]]`, `kwargs_lattice` periódico con
   `order: "default_2"`, `CM.selection: "J1J2Square"` con `J1: 1.0`, `J2_list: [0.5]`.
2. **config2_Hydra.json**: una selección con un único stage, p. ej. ViT2D con proyección de
   momento:
   ```json
   "selection": "ViT2DOH000",
   "ViT2DOH000": {"stage0": {"template": {"Sequivariant": {
       "ViT2D": {}, "OutputHead": {}, "irrep": [0, 0], "phase_mode": "angle", "coset": [2, 2]}}}}
   ```
3. **config3_Hydra_NN.json**: `symm_wrapper.Z2.on: true` (sector par) y en `storage`
   `ViT2D = {"token_size": [2,2], "embedding_d": 60, "n_heads": 10, "n_blocks": 4,
   "n_ffn_layers": 2, "final_architecture": null}`, `OutputHead = {"d_model": 60,
   "only_phase": false}`.
4. **config0.json**: `write_folder_sim` (terminado en `/`), `sz_total: 0`,
   `keeper: true` con su `keeper_setup`, sampler `metropolis` con regla `Exchange`
   (1024 cadenas × 1 muestra, `chunk_vstate: 512`), driver `vmc_sr_netket` y un schedule de
   700 + 300 épocas (§5.1).

Ejecutar:

```bash
./VMC_simulation.py -c config0.json -c config1_CM.json -c config2_Hydra.json -c config3_Hydra_NN.json
```

En pantalla aparecen la configuración resuelta (sampler, optimizador, tabla del schedule,
modelo, stages), el diagrama de la arquitectura con los códigos de cada módulo, el número de
parámetros, la energía exacta (global y del sector) y la barra de progreso con la energía.

## 2.3 Qué se obtiene

En `<write_folder_sim>/J1J2Square/ViT2DOH000_…/Size_4x4/ST_700-A_300-A/UUID_<uuid>/`:

* `…_results_….json`: energía del mejor estado, error frente a ED, fidelidad, V-score,
  observables y rutas de todos los artefactos;
* `Callback_….jpeg`: curva de entrenamiento con la energía exacta;
* `…_best.orbax/`: los mejores estados según cada criterio;
* `…_trace.txt`: energía de cada paso.

Referencia: el estado fundamental exacto 4x4 (J2 = 0.5) es −33.8317 en unidades de Pauli.
Errores típicos de la receta anterior: ~2 % en el GS 4x4 y ~0.5 % en 6x6 (CvT3).

## 2.4 Siguientes pasos

* **Otro sector**: cambie `irrep` (p. ej. `[2, 2]` = (π,π) en 4x4) y el sector Z2
  (`trivial_Z2: false` para impar). El ED del sector se calcula automáticamente.
* **Más rápido**: `"network_dtype": "float32"` en config0 (§5.5).
* **Cambiar de arquitectura o de sector a mitad de simulación**: añada `stage1` en config2 y
  un segundo eon al schedule (§4.5, §5.1).
* **Partir de un resultado anterior**: `load_model` con `load_from` (§4.5).
* **Campañas en cluster**: capítulo 9.
