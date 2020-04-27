import tensorflow as tf


def selected_coil(kspaces):
    n_coils = tf.shape(kspaces)[1]
    i_coil = tf.random.uniform(
        shape=(1,),
        minval=0,
        maxval=n_coils,
        dtype=tf.int32,
    )[0]
    return i_coil
