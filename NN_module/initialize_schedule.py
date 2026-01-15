import jax.numpy as jnp
import optax
import ast


from NN_module.schedule import SCHEDULES
from NN_module.schedule.utils import schedule_from_array, masked_optimizer
from NN_module.ST_utils import print_tree


class Schedule:

    def __init__(self, setup):

        self.epochs_struct = setup["epochs_struct"]
        self.modes_struct = setup["modes_struct"]
        self.lr_struct = setup["lr_struct"]
        self.repeat = setup["repeat"]
        self.rescale = setup["rescale"]

        self._initialize()

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
            assert mode == "B", f"2 or more lr instructions, but mode is {mode}"
            for lr_ins in lr_instruction:
                nruter = self.generate_period(epochs, mode, lr_ins)
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

        for eon_seg, eon_mode, eon_lr in zip(
            self.epochs_struct, self.modes_struct, self.lr_struct
        ):

            for era_seg, era_mode, era_lr in zip(eon_seg, eon_mode, eon_lr):

                for seg, mode, lr in zip(era_seg, era_mode, era_lr):
                    period_array, info = self.generate_period(seg, mode, lr)
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
                    print(info)
                    yield period_array, info

    def schedule(self, return_array=False):

        for period_array, info in self.schedule_generator():

            if return_array:
                yield period_array, info
                continue

            if isinstance(period_array, (list, tuple)):
                period_func = [schedule_from_array(x) for x in period_array]

            yield period_func, info

    def transform_optimizer(self, params, optimizer, lr_func, info):
        mode = info[0][1]

        if len(lr_func) < 2:
            transformation = {
                "freeze": optax.set_to_zero(),
                "train": optimizer(lr_func[0]),
                "train_modulus": optimizer(lr_func[0]),
                "train_phase": optimizer(lr_func[0]),
            }
        else:
            transformation = {
                "freeze": optax.set_to_zero(),
                "train": optimizer(lr_func[0]),
                "train_modulus": optimizer(lr_func[0]),
                "train_phase": optimizer(lr_func[1]),
            }

        trans_tree = masked_optimizer(params, mode=mode)
        trans_optimizer = optax.multi_transform(transformation, trans_tree)
        print(print_tree(trans_tree, values=True))

        return trans_optimizer
