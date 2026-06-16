#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

from time import time
import uuid

import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from einops import rearrange

from NN_module.NN.utils import make_setup_serializable
from NN_module.callback.utils import dump_callback
from NN_module.sim_utils import measureNdump
from NN_module.callback import Callback
from NN_module.label_utils import (
    get_filenames_from_settings,
    get_write_folder_from_model,
    display_simulation_settings,
    get_sim_config,
)
from NN_module.saveNload import save_results

print(jax.devices())

import json
import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    action="append",
    required=False,
    help="Parse configuration files in order. Simulation/CM/NN",
)
args = parser.parse_args()

if args.config is None:
    args.config = [
        "/home/ihuarte/Escritorio/Ivan/NNs/config0.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config1_CM.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config2_Hydra.json",
        "/home/ihuarte/Escritorio/Ivan/NNs/config3_Hydra_NN.json",
    ]
configurations = args.config

print(f"Configurations:")
for c in configurations:
    print(f" - {c}")

# Cargamos configuraciones de archivos json
with open(configurations[0], "r") as f:
    config = json.load(f)
with open(configurations[1], "r") as f:
    config_cm = json.load(f)
with open(configurations[2], "r") as f:
    config_hydra = json.load(f)
with open(configurations[3], "r") as f:
    config_hydra_nn = json.load(f)

config_nn = {**config_hydra, **config_hydra_nn}


cm_model_name = config_cm["CM"]["selection"]
nn_evol_name = config_nn["selection"]

cm_model_setup = config_cm["CM"][cm_model_name]
nn_evol_setup = config_nn[nn_evol_name]

model_label = cm_model_name + "_" + nn_evol_name

sim_config = {
    "SIM": config,
    "CM": cm_model_setup,
    "NN": {"name": nn_evol_name, "setup": nn_evol_setup},
}


def extract_patches2d(x, patch_size):
    batch = x.shape[0]
    n_patches = int((x.shape[1] // patch_size**2) ** 0.5)
    x = x.reshape(batch, n_patches, patch_size, n_patches, patch_size)
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(batch, n_patches, n_patches, -1)
    x = x.reshape(batch, n_patches * n_patches, -1)
    return x


class Embed(nn.Module):
    d_model: int  # dimensionality of the embedding space
    patch_size: int  # linear patch size
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        x = extract_patches2d(x, self.patch_size)
        x = self.embed(x)

        return x


class FactoredAttention(nn.Module):
    n_patches: int  # lenght of the input sequence
    d_model: int  # dimensionality of the embedding space (d in the equations)

    def setup(self):
        self.alpha = self.param(
            "alpha", nn.initializers.xavier_uniform(), (self.n_patches, self.n_patches)
        )
        self.V = self.param(
            "V", nn.initializers.xavier_uniform(), (self.d_model, self.d_model)
        )

    def __call__(self, x):
        y = jnp.einsum("i j, a b, M j b-> M i a", self.alpha, self.V, x)
        return y


from functools import partial


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        if self.transl_invariant:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches),
                self.param_dtype,
            )
            sq_n_patches = int(self.n_patches**0.5)
            assert sq_n_patches * sq_n_patches == self.n_patches
            self.alpha = roll2d(
                self.alpha, jnp.arange(sq_n_patches), jnp.arange(sq_n_patches)
            )
            self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
        else:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

    def __call__(self, x):
        # apply the value matrix in paralell for each head
        v = self.v(x)

        # split the representations of the different heads
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )

        # factored attention mechanism
        v = rearrange(
            v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
        )
        x = jnp.matmul(self.alpha, v)
        x = rearrange(
            x, "batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff"
        )

        # concatenate the different heads
        x = rearrange(
            x, "batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)"
        )

        # the representations of the different heads are combined together
        x = self.W(x)

        return x


class EncoderBlock(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x


log_cosh = (
    nk.nn.activation.log_cosh
)  # Logarithm of the hyperbolic cosine, implemented in a more stable way


class OuputHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j * out_imag

        return jnp.sum(log_cosh(out), axis=-1)


class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    transl_invariant: bool = False

    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)

        Ns = x.shape[-1]  # number of sites
        n_patches = Ns // self.patch_size**2  # lenght of the input sequence

        x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)

        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            transl_invariant=self.transl_invariant,
        )(x)

        log_psi = OuputHead(d_model=self.d_model)(y)

        return log_psi


##### GROUND STATE OPTIMIZATION #####

seed = 0
key = jax.random.key(seed)

# Model
L = 4
n_dim = 2
J2 = 0.5

# sampler and vstate
N_samples = 4096
chunk_size = 4096
learning_rate = 0.01

# ViT
num_layers = 2
d_model = 40
n_heads = 8
patch_size = 2
transl_invariant = True

ds = 1e-4

# Run
epochs = 2000


print(f"L = {L}  || J2 = {J2}")
print(f"Nsamples = {N_samples}  || chunking = {chunk_size}")
print(f"\nViT settings:")
print(f"n_layers: {num_layers}")
print(f"d_model = {d_model}")
print(f"n_heads = {n_heads}")
print(f"patch_size = {patch_size}")
print(f"transl_invariant = {transl_invariant}\n\n")


lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Mar

print("Running exact diagonalization...")
from scipy.sparse.linalg import eigsh

E_ED, x_ED = eigsh(hamiltonian.to_sparse(), k=1, return_eigenvectors=True, which="SA")
# x_ED = full_basis_state(x_ED, hi) if hi._total_sz is not None else x_ED
E_ED = float(E_ED.squeeze(-1))
print(f"Energy ED: {E_ED}")

# Intiialize the ViT variational wave function
vit_module = ViT(
    num_layers=num_layers,
    d_model=d_model,
    n_heads=n_heads,
    patch_size=patch_size,
    transl_invariant=True,
)

key, subkey = jax.random.split(key)
spin_configs = (
    jax.random.randint(subkey, shape=(N_samples, L * L), minval=0, maxval=1) * 2 - 1
)
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
vsampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

voptimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

key, subkey = jax.random.split(key, 2)
vvstate = nk.vqs.MCState(
    sampler=vsampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=chunk_size,
)

N_params = nk.jax.tree_size(vvstate.parameters)
print("Number of parameters = ", N_params, flush=True)
sim_uuid = str(uuid.uuid4())[:8]
write_folder = f"/home/ihuarte/Escritorio/Ivan/NNs/Viteretti_{L}x{L}/UUID_{sim_uuid}/"
os.makedirs(write_folder, exist_ok=True)


# Variational monte carlo driver
from netket._src.driver.vmc_sr import VMC_SR

#### INITIALIZE CALLBACKS ####
cm_model_setup["name"] = "J1J2Square"
cm_model_setup["size"] = [L, L]
cm_model_setup["params"] = {"J1": 1.0, "J2": 0.5, "fields": [0.0, 0.0, 0.0]}
if config["callback"]["checkpoint"]:
    sim_label, _, _, _ = get_filenames_from_settings(
        cm_model_setup, {"name": nn_evol_name, "setup": {}}, sim_uuid
    )
    sim_label_folder = write_folder + sim_label
    do_each_checkpoint = config["callback"]["checkpoint_setup"]["do_each"]
else:
    sim_label_folder = ""
    do_each_checkpoint = None

callback_objects, callback_funcs = Callback(
    config["callback"],
    sim_config=sim_config,
    total_epochs=epochs,
    H=hamiltonian,
    N=L * L,
    E_ED=E_ED,
    x_ED=x_ED,
    sim_label_folder=sim_label_folder,
    sim_folder=write_folder,
)

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=voptimizer,
    diag_shift=ds,
    variational_state=vvstate,
    mode="complex",
)

# Optimization
log = nk.logging.RuntimeLog()

# sys.exit(0)
time_in = time()
vmc.run(n_iter=epochs, out=log, callback=callback_funcs, show_progress=True)
time_out = time()

time_exe = time_out - time_in
print(f"Execution time: {time_exe:.2f} seconds")

energy = (log.data["Energy"]["Mean"].real)[-1]
energy_per_site = energy / (L * L)
var = (log.data["Energy"]["Variance"].real)[-1]
vscore = L * L * var / energy**2
print(f"Energy: {energy}")
print(f"Vscore: {vscore}")
print(f"Energy per site: {energy_per_site}")


sim_label = f"Viteretti_simulation_{L}x{L}_"

log.best_state = vvstate
log.best_step = epochs - 1
log.best_state_energy = energy
log.best_state_vscore = vscore

print(hasattr(log, "E_ED"))

callback_args = {
    "size": [L, L],
    "write_folder": write_folder,
    "time_exe": time_exe,
    "sim_label": sim_label,
    "title_label_callback": f"Viteretti {L} x {L}",
    "best_step": epochs - 1,
    "schedule_setup": None,
    "do_each_checkpoint": None,
}
callback_artifacts = dump_callback(log, callback_args)
## Calculate some observables
results, mp_array_vs, _ = measureNdump(
    log,
    time_exe,
    exact_diag=False,
    S_operators=False,
)

## Save the results
dump_setup = {
    "results": results,
    "optimizer": "Sgd",
    "_artifacts": {"callback": callback_artifacts},
}
dump_setup = make_setup_serializable(dump_setup)

dump_setup = {**dump_setup, "NN": None, "SIM": None, "CM": None}


save_results(
    vvstate,
    dump_setup,
    x_ED=None,
    modphase=mp_array_vs,
    modphase_ED=None,
    write_folder=write_folder,
    sim_label=sim_label,
    ED_label=None,
    json_label="_results_",
    sim_uuid=sim_uuid,
)
