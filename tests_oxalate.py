#!/home/ihuarte/miniconda3/envs/conda_env/bin/python
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import os
import argparse
import json
import sys
import pathlib
import orbax.checkpoint as ocp

from NN_module.observables import modphase
from NN_module.dynamic_plots import plot_modphase
from NN_module.label_utils import get_filenames_from_settings
from NN_module.callback.utils import checkpoint_callback
from NN_module.label_utils import get_filenames_from_settings
from NN_module.saveNload import load_vstate

# Añadir el directorio chebyoxa al path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "chebyoxa"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "ATMOS_VA"))


from VA.src.VA_project.model.model import Oxalate
from VA.src.VA_project.engine.runners import Runner


a = 0.2
theta = 9.0
phi = 72.0

bc="periodic"
order="default_2"

energy_pa = []
energy_pa_ED = []

#### Diagonalizable lattices

for i, size in enumerate([[3, 3], [3, 4], [4,3], [4, 4], [4,5], [5,4]]):
    
    # size = (L, L)    
    model = Oxalate(size, [a, theta, phi], bc=bc, order=order)
    eng = Runner(model.cm, S_operators=False)
    E_ED = eng.exact_energy_lanczos()/(4*size[0]*size[1])
    print(f"Energy from exact diagonalization: {E_ED}")
    energy_pa_ED.append([i, E_ED])


#### ADD VMC results
file_4x4 = "/home/ihuarte/Escritorio/Ivan/NNs/pruebas/Oxalate/ViT2DOH_Sequential_ViT2D_OutputHead/Size_4x4/ST_1000-A_500-A/UUID_4dc094e2/Oxalate_ViT2DOH_results_4x4_strength_0.2_theta_9.0_phi_72.0_date_20260320T003803_UUID_4dc094e2.json"
file_6x6 = "/home/ihuarte/Escritorio/Ivan/NNs/pruebas/Oxalate/ViT2DOHRefineTZ2_Sequential_ViT2D_OutputHead/Size_6x6/ST_2000-A_1000-A/UUID_686ceacf/Oxalate_ViT2DOHRefineTZ2_results_6x6_strength_0.2_theta_9.0_phi_72.0_date_20260330T154956_UUID_686ceacf.json"
file_8x8 = "/home/ihuarte/Escritorio/Ivan/NNs/Simulations/Oxalate/ViT2DOHLongPlay_Sequential_ViT2D_OutputHead/Size_8x8/ST_1000-A_1000-A/UUID_e1778820/Oxalate_ViT2DOHLongPlay_results_8x8_strength_0.2_theta_9.0_phi_72.0_date_20260402T231938_UUID_e1778820.json"
file_10x10 = "/home/ihuarte/Escritorio/Ivan/NNs/Simulations/Oxalate/ViT2DOHLongPlay3GPUReload_Sequential_ViT2D_OutputHead/Size_10x10/ST_100-A_100-A_1500-A/UUID_47ac3c90/Oxalate_ViT2DOHLongPlay3GPUReload_results_10x10_strength_0.2_theta_9.0_phi_72.0_date_20260404T154330_UUID_47ac3c90.json"

#ViT
E_pa = json.load(open(file_4x4))["results"]["E_best_per_site"]
print(f"Energy from VMC: {E_pa}")
energy_pa.append([4, E_pa])

E_pa = json.load(open(file_6x6))["results"]["E_best_per_site"]
print(f"Energy from VMC: {E_pa}")
energy_pa.append([6, E_pa])

E_pa = json.load(open(file_8x8))["results"]["E_best_per_site"]
print(f"Energy from VMC: {E_pa}")
energy_pa.append([8, E_pa])

E_pa = json.load(open(file_10x10))["results"]["E_best_per_site"]
print(f"Energy from VMC: {E_pa}")
energy_pa.append([10, E_pa])

#CvT
file_4x4_CvT = "/home/ihuarte/Escritorio/Ivan/NNs/pruebas/Oxalate/CvTComplexHead_Sequential_CvT_ComplexHead/Size_4x4/ST_1000-A_500-A/UUID_a0596d5f/Oxalate_CvTComplexHead_results_4x4_strength_0.2_theta_9.0_phi_72.0_date_20260330T223116_UUID_a0596d5f.json"
energy_pa_cvt = json.load(open(file_4x4_CvT))["results"]["E_best_per_site"]
print(f"Energy from VMC with CvT: {energy_pa_cvt}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot([L for L, E in energy_pa_ED], [E for L, E in energy_pa_ED], color = "green", ls="", marker="o", ms=7, label='Exact Diagonalization')
plt.plot([L for L, E in energy_pa], [E for L, E in energy_pa], color = "blue", ls="", marker="s", ms=7, label='VMC (ViT)')
plt.plot([4], [energy_pa_cvt], color = "red", ls="", marker="^", ms=7, label='VMC (CvT)')
plt.xlabel('L (Lattice Size = LxL)')
plt.ylabel('Energy per Atom')
plt.legend()
plt.savefig("ZZenergy_comparison.png", dpi=600, bbox_inches='tight')