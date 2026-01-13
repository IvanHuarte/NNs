import jax.numpy as jnp
import ast

from NN_module.schedule import SCHEDULES


class Schedule:

    def __init__(self, setup):

        self.segments = setup["segments"]
        self.modes = setup["modes"]
        self.lr_struct = setup["lr_struct"]
        self.repeat = setup["repeat"]
        self.rescale = setup["rescale"]

    def period_from_string(self, segment, lr_instruction):
        schedule_name, str_args = lr_instruction.split("(")
        schedule = SCHEDULES[schedule_name]
        args = ast.literal_eval("(" + str_args)

        period = schedule(segment, *args)
        return period

    def generate_period(self, segment, mode, lr_instruction):

        if len(lr_instruction) > 1:
            period = []
            info = []
            assert mode == "B", f"2 or more lr instructions, but mode is {mode}"
            period.append(
                self.generate_period(segment, mode, lr_ins) for lr_ins in lr_instruction
            )

        if isinstance(lr_instruction, int):
            period = jnp.array([lr_instruction] * segment)
            info = (segment, mode, lr_instruction)

        elif isinstance(lr_instruction, str):
            period = self.period_from_string(segment, mode, lr_instruction)
            info = (segment, mode, lr_instruction)

        return period, info

    def build_schedule(self):

        periods = []

        for i, (eon_seg, eon_mode, eon_lr) in enumerate(
            zip(self.segments, self.modes, self.lr_struct)
        ):

            for j, (era_seg, era_mode, era_lr) in enumerate(
                zip(eon_seg, eon_mode, eon_lr)
            ):

                for k, (seg, mode, lr) in enumerate(zip(era_seg, era_mode, era_lr)):
                    
                    yield self.generate_period(seg, mode, lr)
                    # lr_period, period_info = self.generate_period(seg, mode, lr)
                    # periods[0].append(lr_period)
                    # periods[1].append(period_info)
