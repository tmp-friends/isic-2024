import tensorflow as tf
from tensorflow.keras import backend as K


def R2Score(y_true, y_pred):
    """
    Calculate R2 for every target seperatly, then take the average.
    """
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)

    R2 = 1 - SS_res / (SS_tot + K.epsilon())

    return K.mean(R2)


@tf.keras.utils.register_keras_serializable(package="MyMetrics", name="ClippedR2Score")
class ClippedR2Score(tf.keras.metrics.Metric):
    def __init__(self, name="r2_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_metric = tf.keras.metrics.R2Score(class_aggregation=None)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.base_metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return tf.reduce_mean(tf.clip_by_value(self.base_metric.result(), 0.0, 1.0))

    def reset_states(self):
        self.base_metric.reset_states()
