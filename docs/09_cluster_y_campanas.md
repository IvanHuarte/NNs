# 9. Clusters y campañas de simulaciones

## 9.1 Máquinas

| Máquina | GPU | float64 | Uso recomendado |
|---|---|---|---|
| Local | RTX 3080 10 GB | 1/64 de float32 | Pruebas, 4x4–6x6, cribados; red en float32 |
| garmonegro | L40S 45 GB (2 por trabajo) | 1/64 de float32 | Trabajos largos y pesados (10x10, 12x12), sin límite de tiempo y con entrada inmediata en cola; red en float32 |
| CESGA | A100 40 GB | Rápido | Trabajos float64 y lotes de trabajos cortos; la cola tarda ~2 días en arrancar |

Rutas raíz: garmonegro `/home/ihuarte/Escritorio/Ivan/NNs/`; CESGA
`/mnt/netapp1/Store_CSIC//home/csic/atm/ihf/Escritorio/Ivan/NNs/`.

## 9.2 Lanzar trabajos con `pile`

`pile` envía un trabajo de Slurm que ejecuta un script del repositorio con sus configs:

```bash
P="-s VMC_simulation.py -c config0.json,config1_CM.json,config2_Hydra.json,config3_Hydra_NN.json -g 2"
pile $P -R <carpeta>          # en los clusters: usa <carpeta>/config_files/
```

| Opción | Significado |
|---|---|
| `-s` | Script |
| `-c` | Configs separados por comas |
| `-g` | Número de GPUs |
| `-p` | Partición (por defecto `gpu`) |
| `-r JID` | Relanzar con los JSON de `pile/JID/` |
| `-R dir` | (versión de los clusters) usar `dir/config_files/` |
| `-d JID` | Dependencia **`afterany`**: arranca aunque el trabajo origen falle |

Cada trabajo crea `pile/<JobId>/` con sus cuatro configs y `logs/{out,error}_<JobId>.log`
(la primera línea de `out` es el comando exacto). **No escriba nada a mano en `pile/`.**

Reglas aprendidas:

* **Ningún trabajo debe depender de otro.** Con `-d` (afterany) un trabajo dependiente
  arranca aunque su origen haya muerto y falla al cargar; además `pile` adivina el JobId.
  Las cadenas de sectores se hacen como stages de Hydra dentro de un único trabajo, y los
  arranques en caliente sólo leen simulaciones ya terminadas.
* CvT3 en 8x8 o más necesita `chunk_size_bwd` (256 con 2048 muestras por GPU).
* Muestras: con `-g 2`, 1024 cadenas × 2 muestras × 2 GPUs = 4096. Con 1 GPU, duplicar
  `n_chains_per_rank` para mantener el total.

### En local

* `Simulations_claude/submit.sh CONFIG_DIR [-g N] [-p partición]` reproduce la
  estructura de `pile/<JobId>/` en el Slurm local sin la carrera del JobId.
* `Simulations_claude/run_local.sh DIR [DIR…]` ejecuta configs en secuencia sin Slurm
  (registro en `Simulations_claude/logs/<nombre>.log`). Ajuste `TMPDIR` en el script a una
  carpeta propia.
* Si el nodo Slurm local aparece como `down*` tras un reinicio:
  `sudo systemctl restart slurmd` y
  `sudo scontrol update nodename=localhost state=resume`.

## 9.3 Generar campañas: `Simulations_claude/gen.py`

`make(...)` parte de los cuatro configs de la raíz (sólo lectura) y devuelve
`(config0, config1, config2, config3)`; `write(raíz, nombre, cfgs)` los escribe.

```python
from gen import make, write
cfgs = make(L=8, net="cvt", net_kw=dict(kernel=[[3, 3], [3, 3]]),
            irrep=(0, 0), coset=(2, 2), z2=+1,
            chains=1024, spc=2, chunk_vstate=2048, chunk_bwd=256,
            epochs=(1400, 600), lr=(0.007, "exp(0.007,0.0005)"),
            diag_shift="sig(0.01,0.0001,800)",
            write_root="/…/Simulations_claude_cluster/8x8", dir_name="c8_CvT3OH000",
            network_dtype="float32")
write("Simulations_claude/mi_campaña", "c8_CvT3OH000", cfgs)
```

| Argumento | Significado |
|---|---|
| `L`, `net` (`"vit"`/`"cvt"`), `net_kw` | Tamaño y red (los `net_kw` sobrescriben los valores base) |
| `irrep`, `coset`, `phase_mode`, `z2` (`None`, `+1`, `−1`) | Sector y proyección |
| `chains`, `spc`, `chunk_vstate`, `chunk_bwd`, `sampler_rules` | Muestreo y memoria |
| `epochs`, `lr`, `diag_shift`, `momentum`, `seed` | Schedule de un solo stage |
| `stages=[dict(irrep=…, coset=…, z2=…, epochs=…, lr=…, diag_shift=…), …]` | Varios stages en un trabajo (cadena de sectores); un cambio de Z2 añade el `symm_wrapper` del stage |
| `load_from`, `load_sampler` | Arranque en caliente |
| `suffix`, `sel` | Nombre de la selección (por defecto `<Módulo>OH<irrep><suffix>`) |
| `keeper_modes`, `keeper_save_every`, `best_c`, `checkpoint_every` | Keeper y checkpoints |
| `network_dtype`, `matmul_precision` | Precisión |

Modelo fijo: J1-J2 con J2 = 0.5, driver `vmc_sr_netket`, keeper
`["best","best_energy","best_vscore"]`.

### Campañas existentes

| Script | Contenido |
|---|---|
| `local_4x4.py`, `local_4x4_r.py`, `local_6x6.py`, `local_cvt.py` | Campañas locales (configs en `Simulations_claude/configs/…`, resultados en `runs/…`) |
| `cluster.py` | Primera ronda de cluster (obsoleta: usaba dependencias) |
| `cluster2.py [--dtype float32] [--host cesga] [nombres]` | Segunda ronda: estados excitados en el subespacio del stride (3a), CvT3 8x8 y cribado de opciones (3c), 10x10 y 12x12 (3b). Escribe `cluster2{_f32}{_cesga}/<c<L>_<sel>>/config_files/` |

`cluster2_order.txt` (garmonegro) y `cluster2_cesga_order.txt` (CESGA) listan los trabajos
en orden de prioridad, relativos a `Simulations_claude/`:

```bash
T=/home/ihuarte/Escritorio/Ivan/NNs/Simulations_claude
for j in $(cat $T/cluster2_order.txt); do pile $P -R $T/$j; done
```

La documentación detallada de cada ronda está en `Simulations_claude/CLUSTER2.md`, y el
diario de pruebas en `Simulations_claude/REPORT_weekend.md`.

## 9.4 Seguimiento

* `tail` de `<sim>_trace.txt` muestra el progreso paso a paso de cualquier trabajo.
* `analyze_cluster2.py <raíz>` resume todos los trabajos (también los muertos o en marcha).
* Una simulación muerta conserva sus mejores estados en `_best.orbax`: se puede continuar
  apuntando `load_from` a su carpeta.
* Estrategia recomendada: explorar en float32 y refinar en float64 (300–500 épocas, lr bajo,
  diag_shift ~1e-4) sólo los casos que necesitan un número final, partiendo de
  `_best.orbax/best`.
