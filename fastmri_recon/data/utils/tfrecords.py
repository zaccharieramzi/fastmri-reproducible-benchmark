import tensorflow as tf


TENSOR_DTYPES = {
    'kspace': tf.complex64,
    'ktraj': tf.float32,
    'output_shape': tf.int32,
    'volume': tf.float32,
    'dcomp': tf.complex64,
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

def feature_decode():
    return tf.io.FixedLenFeature(shape=(), dtype=tf.string)

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
    extra_args = (example_parsed['output_shape'],)
    if compute_dcomp:
        extra_args += (example_parsed['dcomp'])
    model_inputs = (example_parsed['kspace'], example_parsed['ktraj'], extra_args)
    model_outputs = example_parsed['volume']
    return model_inputs, model_outputs

def get_extension_for_acq(
        volume_size=(256, 256, 256),
        scale_factor=1,
        acq_type='radial_stacks',
        **acq_kwargs,
    ):
    if volume_size != (256, 256, 256):
        raise ValueError(f'Volume size should be (256, 256, 256) and is {volume_size}')
    if scale_factor != 1e-2:
        raise ValueError(f'Scale factor should be 1e-2 and is {scale_factor}')
    af = acq_kwargs['af']
    return f'_{acq_type}_af{af}'
