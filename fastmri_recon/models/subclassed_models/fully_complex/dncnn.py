from tensorflow.keras.models import Model
from tf_complex.convolutions import ComplexConv2D


class DnCNN(Model):
    def __init__(
            self,
            n_convs=3,
            n_filters=16,
            n_outputs=1,
            activation='crelu',
            res=True,
            **kwargs,
        ):
        super(DnCNN, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.n_outputs = n_outputs
        self.activation = activation
        self.res = res
        self.convs = [
            ComplexConv2D(
                self.n_filters,
                3,
                padding='same',
                activation=self.activation,
                kernel_initializer='glorot_uniform',
            )
            for i in range(self.n_convs-1)
        ]
        self.convs.append(ComplexConv2D(
            2*self.n_outputs,
            3,
            padding='same',
            activation='linear',
            kernel_initializer='glorot_uniform',
        ))

    def call(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        if self.res:
            outputs = inputs + outputs
        return outputs
