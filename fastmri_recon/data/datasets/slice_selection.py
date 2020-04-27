import tensorflow as tf


def selected_slices(kspaces, inner_slices=8, rand=True):
    n_slices = tf.shape(kspaces)[0]
    slice_start = n_slices // 2 - inner_slices // 2
    if rand:
        i_slice = tf.random.uniform(
            shape=(1,),
            minval=slice_start,
            maxval=slice_start + inner_slices - 1,
            dtype=tf.int32,
        )
        slices = (i_slice[0], i_slice[0] + 1)
    else:
        slices = (slice_start, slice_start + inner_slices)
    return slices
