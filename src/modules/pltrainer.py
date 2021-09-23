from collections import OrderedDict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint


class pltrain():

    def __init__(self, model, args, data_module):
        self.model = model
        self.data_module = data_module
        self.args = args

        # TODO sanity check on the parameters - config
        #gpu = None if not args.get('gpu', False) else args.get('gpu')

        self.checkpoint_callback = ModelCheckpoint(
            monitor="monitor_metrics",
            dirpath = args.record_dir,
            filename="model_best",
            mode="max",
        )

        self.trainer = pl.Trainer(
            #gradient_clip_val=0.1,
            max_epochs=args.epochs, deterministic=True,
            gpus=args.n_gpu,
            plugins=DDPPlugin(find_unused_parameters=False),
            callbacks=[EarlyStopping(monitor="monitor_metrics", mode="max", patience=args.early_stop, verbose=True), self.checkpoint_callback]
        )

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.data_module)

    def evaluate(self):
        self.trainer.test(model=self.model, datamodule=self.data_module)
