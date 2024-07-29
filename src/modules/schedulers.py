import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback


class CosineAnnealer:
    """Cosine annealer for learning rate scheduling"""

    def __init__(self, start, end, steps):
        """
        Args:
            start (float): Initial value.
            end (float): Final value.
            steps (int): Number of steps to anneal.
        """
        self.start = start

        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        """
        Returns:
            float: Annealed value.
        """
        self.n
        cos = np.cos(np.pi * (self.n / self.steps)) + 1

        return self.end + (0.5 * (self.start - self.end) * cos)


class OneCycleScheduler(Callback):
    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.01, div_factor=10.0):
        """
        Args:
            lr_max (float): Maximum learning rate.
            steps (int): Number of steps to anneal.
            mom_min (float): Minimum momentum.
            mom_max (float): Maximum momentum.
        """
        super().__init__()

        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 10)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        self.phases = [
            [CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
            [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)],
        ]
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().steps())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass  # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title("Learning Rate")
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title("Momentum")


def scheduler(epoch, lr):
    if epoch < 1 or epoch > 1:
        return lr
    else:
        return lr / (25 * 10)


def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, start_lr=0.0, target_lr=1e-3):
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
    )

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold, learning_rate, target_lr)

    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)

    return learning_rate
