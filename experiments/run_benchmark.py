import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lib.nn.baselines.diffconv_tts_model import DiffConvTTSModel
from lib.nn.baselines.gated_tts_model import GatedTTSModel
from lib.nn.baselines.gconv_tts_model import GraphConvTTSModel
from lib.nn.hier_predictor import HierPredictor
from lib.nn.baselines.gunet_tts_model import GUNetTTSModel
from lib.nn.hierarchical.models.higp_tts_model import HiGPTTSModel
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.engines import Predictor
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics

from lib.datasets.air_quality import AirQuality

def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == 'gunet_tts':
        model = GUNetTTSModel, Predictor
    elif model_str == 'diff_tts':
        model = DiffConvTTSModel, Predictor
    elif model_str == 'gconv_tts':
        model = GraphConvTTSModel, Predictor
    elif model_str == 'gated_conv_tts':
        model = GatedTTSModel, Predictor
    elif model_str == 'higp_tts':
        model = HiGPTTSModel, HierPredictor
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif name == 'bay':
        dataset = PemsBay()
    elif name == 'air':
        dataset = AirQuality(impute_nans=True)
    elif name == 'cer':
        raise ValueError(f"Request access to the dataset at https://www.ucd.ie/issda/data/commissionforenergyregulationcer/")
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run_traffic(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)

    if cfg.get('mask_as_exog', False) and 'u' in torch_dataset:
        torch_dataset.update_input_map(u=['u', 'mask'])

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis)
    }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice)
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    model_cls, pred_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        window=torch_dataset.window,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mre': torch_metrics.MaskedMRE()}

    if cfg.dataset.name in ['la', 'bay']:
        multistep_metrics = {
            'mape': torch_metrics.MaskedMAPE(),
            'mae@15': torch_metrics.MaskedMAE(at=2),
            'mae@30': torch_metrics.MaskedMAE(at=5),
            'mae@60': torch_metrics.MaskedMAE(at=11),
        }
        log_metrics.update(multistep_metrics)

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    if pred_cls is HierPredictor:
        additional_pred_kwargs = dict(
            forecast_reconciliation=cfg.forecast_reconciliation,
            lam=cfg.lam,
            levels=cfg.model.hparams.levels,
            warm_up=cfg.get('warm_up'),
            reconciliation_start_epoch=cfg.get('reconciliation_start_epoch'),
            beta=cfg.beta
        )
    else:
        additional_pred_kwargs = dict()

    # setup predictor
    predictor = pred_cls(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
        **additional_pred_kwargs
    )

    ########################################
    # logging options                      #
    ########################################

    run_args = exp.get_config_dict()
    run_args['model']['trainable_parameters'] = predictor.trainable_parameters

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    trainer.test(predictor, dataloaders=dm.test_dataloader())

    exp_logger.finalize('success')


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic, config_path='../config/benchmark',
                     config_name='default')
    exp.run()
