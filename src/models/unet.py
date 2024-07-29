import tensorflow as tf


def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv1D(num_filters, kernel_size=3, padding="same", activation="elu")(inputs)
    x = tf.keras.layers.Conv1D(num_filters, kernel_size=3, padding="same", activation="elu")(x)

    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv1DTranspose(num_filters, kernel_size=2, strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)

    return x


def unet_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding1D(padding=2)(inputs)

    # Encoder
    s00, p00 = encoder_block(x, 32)
    s0, p0 = encoder_block(p00, 64)
    s1, p1 = encoder_block(p0, 128)
    s2, p2 = encoder_block(p1, 256)

    # Bottleneck
    b1 = conv_block(p2, 512)

    # Decoder
    d3 = decoder_block(b1, s2, 256)
    d4 = decoder_block(d3, s1, 128)
    d5 = decoder_block(d4, s0, 64)
    d6 = decoder_block(d5, s00, 32)

    # Output
    outputs = tf.keras.layers.Conv1D(6, kernel_size=1, padding="same")(d6)
    outputs = tf.keras.layers.Cropping1D(cropping=2)(outputs)
    model = tf.keras.Model(inputs, outputs, name="U-Net")

    return model
