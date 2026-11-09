# NNs — Neural Quantum States con Monte Carlo variacional

Entorno de simulación de estados cuánticos neuronales (NQS) para modelos de espines en redes
2D, basado en NetKet, JAX y Flax. Calcula estados fundamentales y estados excitados por
sectores de momento y paridad Z2, con validación frente a diagonalización exacta.

* **Configuración declarativa**: cuatro JSON (simulación, modelo, red, hiperparámetros).
* **Hydra**: constructor de redes (ViT2D, CvT3, CNN…) con proyección exacta de momento,
  arquitecturas que evolucionan por etapas y trasplante de pesos.
* **BestIterKeeper**: mejores estados en disco por energía, V-score o criterio conjunto.
* **Precisión mixta** float32/float64 y herramientas para campañas en clusters Slurm.

## Uso rápido

```bash
uv sync
./VMC_simulation.py -c config0.json -c config1_CM.json -c config2_Hydra.json -c config3_Hydra_NN.json
JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests -q
```

## Documentación

El manual completo está en [`docs/`](docs/README.md).
