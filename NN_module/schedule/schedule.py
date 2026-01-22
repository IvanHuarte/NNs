import jax.numpy as jnp
import optax
import flax.linen as nn

from NN_module.schedule.masks import masked_optimizer
from NN_module.schedule.transformations import transformation_dictionary
from NN_module.schedule.utils import (
    schedule_from_array,
    decode_arch_labels,
    get_submodules_dict,
)
from NN_module.ST_utils import print_tree


class Schedule:

    def __init__(self, setup, NN_params):

        self.print_arch = setup["print_arch"]

        # Learning rate setup
        learning_rate = setup["learning_rate"]
        self.epochs_struct = learning_rate["epochs_struct"]
        self.modes_struct = learning_rate["modes_struct"]
        self.lr_struct = learning_rate["lr_struct"]
        self.repeat = learning_rate["repeat"]
        self.rescale = learning_rate["rescale"]

        # Architecture evolution along simulation
        arch_evol = setup["architecture"]

        # Sampler orders
        sampler_evol = setup["sampler"]

        self._initialize()
        self.update_subarch_struct(NN_params)
        print(self.code2path, "\n")
        print(self.path2code, "\n")

    def update_subarch_struct(self, NN_params):
        """
        Necesario cambiarlo si se cambia la arquitectura
        """

        codes = [
            code
            for eon_m in self.modes_struct
            for era_m in eon_m
            for per_m in era_m
            for code in per_m
        ]
        # Create all path codes of architecture
        submod_dict = get_submodules_dict(NN_params, self.print_arch)
        # Conserve only those which appear in the simulation
        self.code2path = {}
        for code in codes:
            if code == "A":
                main_branches = dict(
                    [(k, v) for k, v in submod_dict.items() if len(k) == 1]
                )
                self.code2path.update(**main_branches)
            else:
                self.code2path[code] = submod_dict[code]

        self.path2code = dict([(v, k) for k, v in self.code2path.items()])

    def _initialize(self):

        # Repeat substructures
        for target, n_repeat in self.repeat:
            E = int(target[1])
            e = int(target[3])
            [
                self.epochs_struct[E].insert(e, self.epochs_struct[E][e])
                for _ in range(n_repeat)
            ]
            [
                self.modes_struct[E].insert(e, self.modes_struct[E][e])
                for _ in range(n_repeat)
            ]
            [self.lr_struct[E].insert(e, self.lr_struct[E][e]) for _ in range(n_repeat)]

        flat_epochs = jnp.array(
            [period for eon in self.epochs_struct for era in eon for period in era]
        )
        self.total_epochs = int(flat_epochs.sum())
        self.total_periods = flat_epochs.shape[0]

    def flat_setup(self):
        nruter = {}
        epochs = []
        modes = []
        lr_instructions = []
        for eon_epo, eon_mode, eon_lr in zip(
            self.epochs_struct, self.modes_struct, self.lr_struct
        ):
            for era_epo, era_mode, era_lr in zip(eon_epo, eon_mode, eon_lr):
                for epo, mode, lr in zip(era_epo, era_mode, era_lr):

                    mode = [
                        self.code2path[idx] if isinstance(idx, int) else "A"
                        for idx in mode
                    ]
                    epochs.append(epo)
                    modes.append(mode)
                    lr_instructions.append(lr)

        nruter["epochs"] = epochs
        nruter["modes"] = modes
        nruter["lr_instructions"] = lr_instructions
        nruter["rescale"] = self.rescale

        return nruter

    def schedule_generator(self):

        for eon_epo, eon_mode, eon_lr in zip(
            self.epochs_struct, self.modes_struct, self.lr_struct
        ):
            for era_epo, era_mode, era_lr in zip(eon_epo, eon_mode, eon_lr):
                for epo, mode, lr in zip(era_epo, era_mode, era_lr):
                    print(f"Mode: {mode} LR: {lr}")
                    mode, lr = decode_arch_labels(self.code2path, mode, lr)
                    print(f"Mode: {mode} LR: {lr}")
                    period_array, info = self.generate_period(epo, mode, lr)
                    if not isinstance(period_array, list):
                        period_array = [period_array]
                        info = [info]
                    period_array = [array * self.rescale for array in period_array]
                    info = [
                        (
                            (*inf, f"rescaled by {self.rescale:.2f}")
                            if self.rescale != 1.0
                            else inf
                        )
                        for inf in info
                    ]
                    yield period_array, info

    def schedule(self, return_array=False):

        for period_array, info in self.schedule_generator():

            if return_array:
                yield period_array, info
                continue

            if isinstance(period_array, (list, tuple)):
                period_func = [schedule_from_array(x) for x in period_array]

            yield period_func, info

    def transform_optimizer(self, params, optimizer, info, lr_func):

        modes_path = [inf[1] for inf in info]
        modes_code = [self.path2code[path] for path in modes_path]

        trans_dict = transformation_dictionary(optimizer, modes_code, lr_func)
        trans_tree = masked_optimizer(params, modes_path, self.path2code)
        trans_optimizer = optax.multi_transform(trans_dict, trans_tree)
        # print(print_tree(trans_tree, values=True))

        return trans_optimizer
