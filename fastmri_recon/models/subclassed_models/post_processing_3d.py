from tensorflow.keras.models import Model

from fastmri_recon.models.subclassed_models.vnet import Vnet


class PostProcessVnet(Model):
    def __init__(self, recon_model, vnet_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.recon_model = recon_model
        self.recon_model.trainable = False
        self.postproc_model = Vnet(post_processing=True, **vnet_kwargs)

    def call(self, inputs):
        outputs = self.recon_model(inputs)
        outputs = outputs[None]
        outputs = self.postproc_model(outputs)
        outputs = outputs[0]
        return outputs

    def predict_batched(self, inputs, batch_size=1):
        outputs = self.recon_model.predict(inputs, batch_size=batch_size)
        outputs = outputs[None]
        outputs = self.postproc_model.predict_on_batch(outputs)
        outputs = outputs[0]
        return outputs
