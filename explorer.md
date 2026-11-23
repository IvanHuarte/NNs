# Explorer: MPS y tensor networks sobre ViT2D

## Directriz del usuario (22 nov 2026, literal)

> hasta nuevo aviso. Quiero hacer funcionar la MPS y que de mejores resultados que OH. Para ello, quiero que consultes publicaciones in the state of the art en las que utilicen arquitecturas basadas en tensor networks, implementes las que consideres oportunas y hagas pruebas en local utilizando el modelo J1J2 en 0.5 para redes 4x4 y 6x6, utilizando para ello la ViT2D como backbone. Quiero que dividas la tarea en 4 agentes: El agente "Papers" se encargara de buscar en la literatura publicaciones sobre NQS en las que se utilicen MPS o Tensor Networks. Tambien pueden ser de interes aquellas publicaciones que utilicen Tensor Networks que se puedan convertir a objetos variacionales tal y como hacemos con la MPS actual. Este agente, tomara la referencia y la implementara de forma modularizada junto al resto de modulos del proyecto bajo la supervision del agente "Optimo". Segundo el agente "Optimo", el cual asegura que las arquitecturas que se implementan lo hacen de la forma mas eficiente posible mediante unit tests en cpu. El tercer agente "Simulater" se encarga de hacer fine tuning de los hiperparametros de la red o del resto de cada simulacion. Tras cualquier simulacion, Simulater analizara los resultados y buscara mediante un razonamiento logico, combinaciones de hiperparametros que puedan dar mejores resultados. Este es un analogo a la tarea principal a la que te estas dedicando hasta ahora. Por ultimo, el agente "Reformer", se encarga de desbloquear situaciones de callejones sin salida en los que pueda encontrarse el agente "Simulater", es decir, que tras muchos intentos (30 diferentes) de diferentes combinaciones de hiperparametros no se logran buenos resultados. El agente Reformer tomara dicha arquitectura y propondra variantes, no ya en el sentido de hiperparametros, activaciones, normas, etc.. sino en el propio flujo de la red, proponiendo cambios drasticos en la estructura de la arquitectura, pudiendose apoyar en la literatura para ello. Comienza por hacer fine tuning a la MPS actual mientras el agente Paper busca nuevos candidatos y actualiza siempre 1 vez cada 2 horas. Quiero que ademas guardes esta directriz que te acabo de dar como explorer.md en el directorio del proyecto. Empieza, buena suerte.

Objetivo general del trabajo (mensaje anterior del usuario): obtener la mejor energía posible con el menor vscore posible. Los trabajos del benchmark a menudo no publican σ² ni vscore, ni dicen si reevalúan la red.

## Organización

- **Coordinador** (sesión principal de Claude): reparte el trabajo entre los agentes y pasa el resultado de uno al siguiente. Informa al usuario cada 2 horas.
- **Papers**: busca en la literatura NQS con MPS o tensor networks, elige candidatos y los implementa como módulos nuevos de Hydra.
- **Optimo**: tests unitarios en CPU de corrección, entrenabilidad y velocidad. Revisa la MPS actual y cada módulo que implemente Papers.
- **Simulater**: ajuste de hiperparámetros en local (4x4 para descartar rápido, 6x6 para confirmar) y análisis de cada tanda.
- **Reformer**: entra tras 30 intentos de Simulater sin superar a OH, y propone cambios en la estructura de la red.

## Archivos (en `Simulations_claude/explorer/`, ignorado por git)

- `attempts.md`: un intento de Simulater por línea; es el contador de los 30.
- `papers.md`: candidatos de la literatura y módulos implementados.
- `optimo.md`: resultados de los tests y medidas de rendimiento.
- `configs/`, `runs/`, `logs/`: las simulaciones locales.

## Lanzamiento de simulaciones (directriz del usuario, 22 nov)

Simulater lanza todas las simulaciones locales con slurm y el comando `pile`, igual que en garmonegro. Nada de `nohup` ni de procesos sueltos en la GPU.

    cd /home/ihuarte/Escritorio/Ivan/NNs
    pile -s VMC_simulation.py -c config0.json,config1_CM.json,config2_Hydra.json,config3_Hydra_NN.json -g 1 -R <carpeta del experimento>

- pile imprime el JobId y crea `pile/<JobId>/`, con la copia de los configs y los logs (`logs/out_<JobId>.log` y `logs/error_<JobId>.log`). Es lo único que se crea en `pile/`, y lo crea pile.
- Un `pile` cada vez, porque pile deduce el JobId siguiente. La cola de slurm (`squeue`) reparte la GPU entre los trabajos de Simulater y los del usuario.
- Cada intento de `attempts.md` lleva su JobId.

## Referencias (J1-J2, J2/J1 = 0.5, estado fundamental, energías totales en unidades de Pauli)

| Red | Exacta | OH actual (ViT2D, tokens 2x2) |
|---|---|---|
| 4x4 | −33.8317 (−0.528620 por sitio) | R1 `runs/4x4/R1_gs_coset_z2`: −33.089 |
| 6x6 | −72.549 (−0.503810 por sitio) | H1 `runs/6x6/H1_gs_vit_coset_z2`: −72.153 |
