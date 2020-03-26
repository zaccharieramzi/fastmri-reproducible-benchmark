from tensorflow.keras.layers import LeakyReLU


def lrelu(x):
    return LeakyReLU(alpha=0.1)(x)
