import tensorflow as tf


TENSOR_DTYPES = {
    'kspace': tf.complex64,
    'ktraj': tf.float32,
    'output_shape': tf.int32,
    'gt_shape': tf.int32,
    'volume': tf.float32,
    'reconstructed_volume': tf.float32,
    'dcomp': tf.complex64,
    'smaps': tf.complex64,
}

## From
# https://github.com/tensorflow/tensorflow/issues/29877#issuecomment-584851380
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_tensor(tensor):
    if isinstance(tensor, tuple):
        return tuple(serialize_tensor(t) for t in tensor)
    else:
        return bytes_feature(tf.io.serialize_tensor(tensor).numpy())

def feature_decode():
    return tf.io.FixedLenFeature(shape=(), dtype=tf.string)

def set_shapes(data_dict):
    for k, v in data_dict.items():
        if k == 'kspace':
            data_dict[k] = tf.reshape(v, [1, 1, -1, 1])
        elif k == 'dcomp':
            data_dict[k] = tf.reshape(v, [1, 1, -1])
        elif k == 'ktraj':
            v.set_shape([1, 3, None])
        elif 'volume' in k:
            v.set_shape([None, None, None, 1])
    return data_dict

def set_shapes_ncmc(data_dict):
    for k, v in data_dict.items():
        if k == 'kspace':
            v.set_shape([None, None, None, 1])
        elif k == 'dcomp':
            v.set_shape([None, None])
        elif k == 'ktraj':
            v.set_shape([None, 2, None])
        elif 'volume' in k:
            v.set_shape([None, None, None, 1])
        elif 'smaps' in k:
            v.set_shape([None, None, None, None])
    return data_dict

# Post-proc functions
def encode_postproc_example(model_inputs, model_outputs):
    model_inputs = [serialize_tensor(mi) for mi in model_inputs]
    model_outputs = [serialize_tensor(mo) for mo in model_outputs]

    feature = {
        'reconstructed_volume': model_inputs[0],
        'volume': model_outputs[0],
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def decode_postproc_example(raw_record):
    features = {
        'reconstructed_volume': feature_decode(),
        'volume': feature_decode(),
    }
    example = tf.io.parse_example(raw_record, features=features)
    example_parsed = {
        k: tf.io.parse_tensor(tensor, TENSOR_DTYPES[k])
        for k, tensor in example.items()
    }
    example_parsed = set_shapes(example_parsed)
    model_inputs = example_parsed['reconstructed_volume']
    model_outputs = example_parsed['volume']
    return model_inputs, model_outputs

# Non cartesian multicoil functions
def encode_ncmc_example(model_inputs, model_outputs):
    model_inputs = [serialize_tensor(mi) for mi in model_inputs]
    model_outputs = [serialize_tensor(mo) for mo in model_outputs]

    feature = {
        'kspace': model_inputs[0],
        'ktraj': model_inputs[1],
        'smaps': model_inputs[2],
        'output_shape': model_inputs[3][0],
        'dcomp': model_inputs[3][1],
        'volume': model_outputs[0],
    }
    if len(model_inputs) == 5:
        # there are different shapes at play here
        # the output_shape is a misnomer and it's actually used inside the
        # network for the nufft computations to have an image that's reduced to
        # its essential components
        # the gt_shape is used for braind data which doesn't have a normalized
        # "output shape" like the knee data
        feature.update(gt_shape=model_inputs[4])
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def decode_ncmc_example(raw_record, slice_random=True, brain=False):
    features = {
        'kspace': feature_decode(),
        'ktraj': feature_decode(),
        'smaps': feature_decode(),
        'output_shape': feature_decode(),
        'dcomp': feature_decode(),
        'volume': feature_decode(),
    }
    if brain:
        features.update(gt_shape=feature_decode())
    example = tf.io.parse_example(raw_record, features=features)
    example_parsed = {
        k: tf.io.parse_tensor(tensor, TENSOR_DTYPES[k])
        for k, tensor in example.items()
    }
    example_parsed = set_shapes_ncmc(example_parsed)
    num_slices = tf.shape(example_parsed['volume'])[0]
    slice_start = tf.random.uniform([1], maxval=num_slices, dtype=tf.int32)[0] if slice_random else 0
    num_slices_selected = 1 if slice_random else num_slices
    slice_end = slice_start + num_slices_selected
    extra_args = (
        example_parsed['output_shape'][slice_start:slice_end],
        example_parsed['dcomp'][slice_start:slice_end],
    )
    model_inputs = (
        example_parsed['kspace'][slice_start:slice_end],
        example_parsed['ktraj'][slice_start:slice_end],
        example_parsed['smaps'][slice_start:slice_end],
    )
    if brain:
        model_inputs += (example_parsed['gt_shape'][slice_start:slice_end],)
    model_inputs += (extra_args,)
    model_outputs = example_parsed['volume'][slice_start:slice_end]
    return model_inputs, model_outputs

# OASIS functions
def encode_example(model_inputs, model_outputs, compute_dcomp=False):
    model_inputs = [serialize_tensor(mi) for mi in model_inputs]
    model_outputs = [serialize_tensor(mo) for mo in model_outputs]

    feature = {
        'kspace': model_inputs[0],
        'ktraj': model_inputs[1],
        'output_shape': model_inputs[2][0],
        'volume': model_outputs[0],
    }
    if compute_dcomp:
        feature['dcomp'] = model_inputs[2][1]
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def decode_example(raw_record, compute_dcomp=False):
    features = {
        'kspace': feature_decode(),
        'ktraj': feature_decode(),
        'output_shape': feature_decode(),
        'volume': feature_decode(),
    }
    if compute_dcomp:
        features['dcomp'] = feature_decode()
    example = tf.io.parse_example(raw_record, features=features)
    example_parsed = {
        k: tf.io.parse_tensor(tensor, TENSOR_DTYPES[k])
        for k, tensor in example.items()
    }
    example_parsed = set_shapes(example_parsed)
    extra_args = (example_parsed['output_shape'],)
    if compute_dcomp:
        extra_args += (example_parsed['dcomp'],)
    model_inputs = (example_parsed['kspace'], example_parsed['ktraj'], extra_args)
    model_outputs = example_parsed['volume']
    model_outputs = model_outputs[None]
    return model_inputs, model_outputs


ACCEPTED_VOLUME_SIZES = [(176, 256, 256), (256, 256, 256)]
def get_extension_for_acq(
        volume_size=(256, 256, 256),
        scale_factor=1,
        acq_type='radial_stacks',
        **acq_kwargs,
    ):
    if volume_size not in ACCEPTED_VOLUME_SIZES:
        raise ValueError(f'Volume size should be in {ACCEPTED_VOLUME_SIZES} and is {volume_size}')
    if scale_factor != 1e-2:
        raise ValueError(f'Scale factor should be 1e-2 and is {scale_factor}')
    af = acq_kwargs['af']
    return f'_{acq_type}_af{af}'
