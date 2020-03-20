from tensorflow.keras.utils import Sequence


# functions from https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
# not ported to tf 2.0 (or at least not found easily)
def iter_sequence_infinite(seq):
    """Iterate indefinitely over a Sequence.
    # Arguments
        seq: Sequence object
    # Returns
        Generator yielding batches.
    """
    while True:
        for item in seq:
            yield item


def is_sequence(seq):
    """Determine if an object follows the Sequence API.
    # Arguments
        seq: a possible Sequence object
    # Returns
        boolean, whether the object follows the Sequence API.
    """
    # TODO Dref360: Decide which pattern to follow. First needs a new TF Version.
    return (getattr(seq, 'use_sequence_api', False)
            or set(dir(Sequence())).issubset(set(dir(seq) + ['use_sequence_api'])))

# from https://github.com/keras-team/keras/blob/master/keras/utils/metrics_utils.py
def to_list(x):
    if isinstance(x, list):
        return x
    return [x]
