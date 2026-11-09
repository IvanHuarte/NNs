# 10. Entorno, desarrollo y tests

## 10.1 Entorno

* Python 3.12 en `.venv` (gestionado con `uv`; `uv sync` lo recrea a partir de
  `pyproject.toml`/`uv.lock`). `requirements.txt` está desactualizado.
* Versiones: jax/jaxlib 0.11.0 (CUDA 13), netket 3.22.3, flax 0.12.8, optax 0.2.8,
  orbax-checkpoint 0.11.39, numpy 2.3.5, scipy 1.17.0.
* `Hydra` y `CouplingModel` no se instalan: se importan desde la raíz del repositorio (los
  scripts de la raíz llevan el shebang `.venv/bin/python`).
* `VMC_simulation.py` fuerza la GPU, desactiva la preasignación de memoria de XLA y activa
  float64 (`jax_enable_x64`).
* Si la compilación en GPU falla con `ptxas … could not be opened`, defina `TMPDIR` a una
  carpeta escribible.
* En clusters, `sbatch_runner` activa `.venv` (o el entorno conda `conda_env`).

## 10.2 Estructura del código

```
VMC_simulation.py        punto de entrada
CouplingModel/           redes (lattice/) y Hamiltonianos (model/), ModelFactory
Hydra/
  NN/                    constructor de redes: Hydra.py, NeuralNetwork.py, utils.py,
                         toolbox.py, precision.py, arch_diagram.py
    NN/FluxModules/      Sequential, Sequivariant, SingleModule, SplitTraining, Transversal
    NN/SingleModules/    ViT2D, CvT3, CvT…, CNN…, MLP, cabezas (FluxFunc.py), fases
    NN/Symm/             Z2, Traslation
    ARNN/                redes autorregresivas (ARViT)
  schedule/              Schedule, funciones lin/exp/sig, máscaras, optimizador
  sampler/               SamplerFactory, reglas, estados iniciales
  VMC/                   VMCBuilder y drivers
  callback/              BestIterKeeper, Checkpoint, plotters, SanityMonitor, LeavesGradient
  exact_diagonalization.py, observables.py, correlations.py
  label_utils.py         nombres de carpetas y ficheros
  saveNload.py           escritura del JSON y recarga de estados
  display.py, dynamic_plots.py, utils.py
pytests/                 tests
Simulations_claude/      generadores y documentación de campañas (no versionado)
docs/                    este manual
```

Principio de diseño: **todo nuevo elemento debe poder controlarse desde los cuatro JSON**
(registros de clases y claves de configuración, no constantes en el código).

## 10.3 Tests

```bash
JAX_PLATFORMS=cpu TMPDIR=<carpeta> .venv/bin/python -m pytest pytests -q        # 145 tests, ~2.5 min
JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_<área> -q              # un área
```

Ejecute siempre `python -m pytest` desde la raíz (algunos tests importan desde el directorio
actual).

| Módulo | Qué verifica |
|---|---|
| `test_arnn` | ARViT y ARNNWrapper: autorregresividad, normalización, muestreo directo |
| `test_callbacks/test_energy_from_log` | Los callbacks usan la energía del log (igual a ⟨H⟩) |
| `test_callbacks/test_keeper_orbax` | Modos del keeper en disco, restauración, stages, `start_fraction`, carga desde `_best.orbax`, puntuación conjunta, traza |
| `test_coupling_model` | J1-J2 frente a Heisenberg de NetKet, unidades, barridos |
| `test_depthwise` | Convolución depthwise circular, `jacrev` bajo la malla de NetKet |
| `test_diagram` | Diagrama de arquitectura |
| `test_equivariance` | Equivariancia de CNNLiang, CvT, CvT3 (todas las opciones) y ViT2D |
| `test_hydra/test_multistage` | Stages, trasplante, `change_attr`, envoltorios activados en stages posteriores |
| `test_irreps` | ED por sectores frente al proyector denso; modos de Sequivariant y coset |
| `test_labels` | Etiquetas estables, `arch_label`, `experiment_name` |
| `test_sampler` | Resolución del bloque del sampler, reglas, estados iniciales |
| `test_schedule` | Códigos de submódulo, modos, diag_shift por eon |

## 10.4 Convenciones

* Nombres de selección: `<Módulo><Cabeza><n_y n_x z2>` + sufijo (`CvT3OH221`).
* Unidades de energía por defecto: Pauli.
* Ejes de la red: `(L0, L1)` = (y, x); irreps `(n_y, n_x)`.
* Las claves de configuración que no son `selection` son de uso personal: guardar
  variantes en los JSON no afecta a la simulación.
