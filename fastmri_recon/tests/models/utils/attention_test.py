import pytest
import tensorflow as tf

from fastmri_recon.models.utils.attention import ChannelAttentionBlock


@pytest.mark.parametrize('kwargs',[
    {},
    {'activation': 'prelu'},
    {'activation': 'lrelu'},
    {'dense': True},
    {'dense': True, 'activation': 'prelu'},
])
def test_channel_attention_block_call(kwargs):
    ca_block = ChannelAttentionBlock(**kwargs)
    inputs = tf.random.normal([2, 32, 32, 16])
    ca_block(inputs)
