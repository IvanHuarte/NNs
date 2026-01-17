import jax.numpy as jnp
import optax
import flax.linen as nn
import ast


from NN_module.schedule import SCHEDULES
from NN_module.schedule.masks import masked_optimizer
from NN_module.schedule.transformations import transformation_dictionary
from NN_module.schedule.utils import schedule_from_array, decode_arch_labels
from NN_module.ST_utils import print_tree


class Schedule:

    def __init__(self, setup, NN_model):
        
        self.arch_name = type(NN_model).__name__
        self.subarch_names = self.subarch_struct(NN_model, self.arch_name)
        print(self.subarch_names)

        self.epochs_struct = setup["epochs_struct"]
        self.modes_struct = setup["modes_struct"]
        self.lr_struct = setup["lr_struct"]
        self.repeat = setup["repeat"]
        self.rescale = setup["rescale"]

        self._initialize()
    
    def subarch_struct(self, NN_model, arch_name):

        if arch_name == "SplitTraining":
            subarch_names = [
                name for name in NN_model.__dict__.keys() if isinstance(
                    NN_model.__dict__[name], nn.Module
                )
            ]
        elif arch_name == "Sequential":
            modules = NN_model.__dict__["Seq"]
            subarch_names = [
                f"Seq_{i}" for i in range(len(modules))
            ]
            subarch_names += ["End"]
        elif arch_name == "Transversal":
            modules = NN_model.__dict__["Trans"]
            subarch_names = [
                f"Trans_{i}" for i in range(len(modules))
            ]
        return subarch_names

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
        epochs = [] ; modes = [] ; lr_instructions = []
        for eon_epo, eon_mode, eon_lr in zip(
            self.epochs_struct, self.modes_struct, self.lr_struct
        ):
            for era_epo, era_mode, era_lr in zip(eon_epo, eon_mode, eon_lr):
                for epo, mode, lr in zip(era_epo, era_mode, era_lr):
                    
                    mode = [
                        self.subarch_names[idx] if isinstance(idx, int) else "A" for idx in mode 
                    ]
                    epochs.append(epo)
                    modes.append(mode)
                    lr_instructions.append(lr)

        nruter['epochs'] = epochs
        nruter['modes'] = modes
        nruter['lr_instructions'] = lr_instructions
        nruter['rescale'] = self.rescale
        
        return nruter

    def period_from_string(self, epochs, lr_instruction):
        schedule_name, str_args = lr_instruction.split("(")
        schedule = SCHEDULES[schedule_name]
        args = ast.literal_eval("(" + str_args)

        period = schedule(epochs, *args)
        return period

    def generate_period(self, epochs, mode, lr_instruction):

        if isinstance(lr_instruction, (list, tuple)):
            period = []
            info = []
            for lr_ins, mod in zip(lr_instruction, mode):
                nruter = self.generate_period(epochs, mod, lr_ins)
                period.append(nruter[0])
                info.append(nruter[1])

        elif isinstance(lr_instruction, (int, float)):
            period = jnp.array([lr_instruction] * epochs)
            info = (epochs, mode, lr_instruction)

        elif isinstance(lr_instruction, str):
            period = self.period_from_string(epochs, lr_instruction)
            info = (epochs, mode, lr_instruction)

        else:
            raise TypeError(f"Unsupported lr_instruction: {lr_instruction}")

        return period, info

    def schedule_generator(self):

        for eon_epo, eon_mode, eon_lr in zip(
            self.epochs_struct, self.modes_struct, self.lr_struct
        ):

            for era_epo, era_mode, era_lr in zip(eon_epo, eon_mode, eon_lr):

                for epo, mode, lr in zip(era_epo, era_mode, era_lr):
                    mode, lr = decode_arch_labels(self.subarch_names, mode, lr)
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

        modes = [inf[1] for inf in info]

        trans_dict = transformation_dictionary(optimizer, modes, lr_func)
        trans_tree = masked_optimizer(params, modes)
        trans_optimizer = optax.multi_transform(trans_dict, trans_tree)
        # print(print_tree(trans_tree, values=True))

        return trans_optimizer
