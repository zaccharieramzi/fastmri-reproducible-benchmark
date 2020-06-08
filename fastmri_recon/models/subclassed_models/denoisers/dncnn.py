from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D


class DnCNN(Model):
    def __init__(
            self,
            n_convs=3,
            n_filters=16,
            n_output_channels=1,
            activation='relu',
            res=True,
            **kwargs,
        ):
        super(DnCNN, self).__init__(**kwargs)
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.n_output_channels = n_output_channels
        self.activation = activation
        self.res = res
        # TODO: maybe have a way to specify non linearity
        self.convs = [
            Conv2D(
                self.n_filters,
                3,
                padding='same',
                activation=self.activation,
                kernel_initializer='glorot_uniform',
            )
            for i in range(self.n_convs-1)
        ]
        self.convs.append(Conv2D(
            self.n_output_channels,
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
