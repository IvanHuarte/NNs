import jax.numpy as jnp
import optax

from NN_module.schedule.masks import masked_optimizer
from NN_module.schedule.optimizer import get_transformed_optimizer
from NN_module.schedule.utils import (
    decode_arch_labels,
    eon_change,
    generate_period,
    period_from_string,
    schedule_from_array,
)


class Schedule:

    def __init__(self, setup, code2path):

        self.eon = 0

        # Learning rate setup
        learning_rate = setup["learning_rate"]
        self.epochs_struct = learning_rate["epochs_struct"]
        self.modes_struct = learning_rate["modes_struct"]
        self.lr_struct = learning_rate["lr_struct"]
        self.repeat = learning_rate["repeat"]
        self.rescale = learning_rate["rescale"]

        # Sampler orders
        # sampler_evol = setup["sampler"]

        # Diag_shift setup
        self.ds_setup = setup["diag_shift"]

        self._initialize()
        self.update_eon_code2path(code2path)

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

    def update_eon_code2path(self, code2path):
        """
        Update eon_code2path and eon_path2code when the architecture changes.
        """
        eon_m = self.modes_struct[self.eon]

        codes = [code for era_m in eon_m for per_m in era_m for code in per_m]

        # From code2path Conserve only those which appear in the simulation
        self.eon_code2path = {}
        for code in codes:
            if code == "A":
                main_branches = dict(
                    [(k, v) for k, v in code2path.items() if len(k) == 1]
                )
                self.eon_code2path.update(**main_branches)
            else:
                self.eon_code2path[code] = code2path[code]

        self.eon_path2code = dict([(v, k) for k, v in self.eon_code2path.items()])

    def flat_setup(self):
        nruter = {}
        epochs = []
        modes = []
        lr_instructions = []
        eons_bound = []
        count = 0
        eon = 0
        for i_eon, (eon_epo, eon_mode, eon_lr) in enumerate(
            zip(self.epochs_struct, self.modes_struct, self.lr_struct)
        ):
            for era_epo, era_mode, era_lr in zip(eon_epo, eon_mode, eon_lr):
                for epo, mode, lr in zip(era_epo, era_mode, era_lr):
                    if eon < i_eon:
                        eons_bound.append(count)
                        eon += 1
                    count += epo
                    mode = [
                        self.eon_code2path[idx] if isinstance(idx, int) else "A"
                        for idx in mode
                    ]
                    epochs.append(epo)
                    modes.append(mode)
                    lr_instructions.append(lr)

        eons_bound.append(count)

        nruter["epochs"] = epochs
        nruter["modes"] = modes
        nruter["lr_instructions"] = lr_instructions
        nruter["rescale"] = self.rescale
        nruter["eons_bound"] = eons_bound

        return nruter

    def locate_period(self, epoch):
        cum = 0
        for i_eon, eon in enumerate(self.epochs_struct):
            for i_era, era in enumerate(eon):
                for i_per, per in enumerate(era):
                    cum += per
                    if cum > epoch:
                        return i_eon, i_era, i_per
        raise ValueError(f"Epoch ({epoch}) out of schedule")

    def schedule_generator(self):

        n_eons = len(self.epochs_struct)

        for i_eon, (eon_epo, eon_mode, eon_lr) in enumerate(
            zip(self.epochs_struct, self.modes_struct, self.lr_struct)
        ):
            n_eras = len(eon_epo)

            for i_era, (era_epo, era_mode, era_lr) in enumerate(
                zip(eon_epo, eon_mode, eon_lr)
            ):
                n_pers = len(era_epo)

                for i_per, (epo, mode, lr) in enumerate(zip(era_epo, era_mode, era_lr)):

                    mode, lr = decode_arch_labels(self.eon_code2path, mode, lr)
                    period_array, info = generate_period(epo, mode, lr)
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
                    change = eon_change(i_eon, i_era, i_per, n_eons, n_eras, n_pers)
                    yield period_array, info, change

    def schedule(self, return_array=False):

        for period_array, info, change in self.schedule_generator():

            if return_array:
                yield period_array, info
                continue

            if isinstance(period_array, (list, tuple)):
                period_func = [schedule_from_array(x) for x in period_array]

            yield period_func, info, change

    def get_diag_shift_schedule(self):


        flat_periods = self.flat_setup()["epochs"]
        main_ds_array = period_from_string(self.total_epochs, self.ds_setup)

        schedule_func = self.ds_setup.split("(")[0]


        ds_periods = [schedule_from_array(main_ds_array[:flat_periods[0]])]
        ds_periods_info = [schedule_func + f"({main_ds_array[0]:.4f}, {main_ds_array[flat_periods[0]]:.4f})"]
        for i in range(1, len(flat_periods)):
            final_epoch = sum(flat_periods[:i])
            ds = schedule_from_array(main_ds_array[final_epoch:final_epoch + flat_periods[i]])
            ds_periods.append(ds)
            ds_periods_info.append(schedule_func + f"({main_ds_array[flat_periods[i-1]]:.4f}, {main_ds_array[flat_periods[i]]:.4f})")

        return ds_periods, ds_periods_info


    def transform_optimizer(self, params, optimizer_setup, info, lr_func):

        modes_path = [inf[1] for inf in info]
        modes_code = [self.eon_path2code[path] for path in modes_path]

        trans_dict = get_transformed_optimizer(optimizer_setup, modes_code, lr_func)
        trans_tree = masked_optimizer(params, modes_path, self.eon_path2code)

        trans_optimizer = optax.multi_transform(trans_dict, trans_tree)

        clip_by_global_norm = optimizer_setup["modifications"]["clip_by_global_norm"]
        if clip_by_global_norm is not None:
            trans_optimizer = optax.chain(
                trans_optimizer, optax.clip_by_global_norm(clip_by_global_norm)
            )
        # print(print_tree(trans_tree, values=True))

        return trans_optimizer
