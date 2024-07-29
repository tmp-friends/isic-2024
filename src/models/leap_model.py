import tensorflow as tf


class LEAPModel(tf.keras.Model):
    def __init__(self, unet):
        super().__init__()

        self.unet = unet
        # self.hidden_size = len(feat_cols) + len(target_cols)
        self.hidden_size = 556 + 368

        # 1. Define the layers
        # self.inp_shape = len(feat_cols)
        self.inp_shape = 556
        # 2. Suggest values of hyperparameters using a trial object
        self.n_layers = 6
        self.activation_fn = "elu"

        inputs = tf.keras.Input(shape=(self.inp_shape,), name="input")
        x = inputs
        x = tf.keras.layers.Dense(3 * self.hidden_size)(x)
        x1 = tf.keras.layers.Dense(2 * self.hidden_size)(x)
        for i in range(self.n_layers):
            x = tf.keras.layers.Dense((2 * self.hidden_size))(x)
            if self.activation_fn == "relu":
                x = tf.keras.layers.ReLU()(x)
            elif self.activation_fn == "elu":
                x = tf.keras.layers.ELU()(x)
            elif self.activation_fn == "leakyRelu":
                x = tf.keras.layers.LeakyReLU(alpha=0.15)(x)

            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.BatchNormalization()(x)

            if (i + 1) % 2 == 0:
                x = tf.keras.layers.add([x, x1])
                x1 = x

        x = tf.keras.layers.Dense(3 * self.hidden_size, activation="elu")(x)
        outputs = tf.keras.layers.Dense(368 - (60 * 6), activation="linear")(x)

        self.ann = tf.keras.Model(inputs, outputs)

    def call(self, x):
        x1 = x[:, 0:360]
        x2 = x[:, 376:]
        x3 = x[:, 360:376]
        x3 = tf.keras.layers.Reshape((1, 16))(x3)
        x3 = tf.keras.layers.Concatenate(axis=1)([x3 for i in range(60)])
        x1 = tf.keras.layers.Reshape((60, 6))(x1)
        x2 = tf.keras.layers.Reshape((60, 3))(x2)
        cnn_input = tf.keras.layers.Concatenate(axis=-1)([x1, x3, x2])

        cnn_out = self.unet(cnn_input)
        ann_out = self.ann(x)
        cnn_out = tf.keras.layers.Flatten()(cnn_out)
        x = tf.keras.layers.Concatenate()([cnn_out, ann_out])

        return x
