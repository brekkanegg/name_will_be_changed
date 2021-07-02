"""
1. pytorch naive reproduce [x]
2. apply Pytorch-Lightning []
3. apply mlflow
4. apply torch-serve
"""

import argparse
import os
import torch
import pytorch_lightning as pl
import torch.multiprocessing


from callbacks import visualizer


def main(cfg):
    print(cfg)

    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor=cfg.metric,
        patience=10,
        verbose=True,
        mode=cfg.metric_mode,
    )

    # FIXME: DDP, SWA 일 떄 뭔가 이상하다.. 이름 설정이 안 먹힘
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=cfg.metric,
        dirpath=cfg.weights_save_path,
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode=cfg.metric_mode,
        filename="best",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    callbacks = None
    callbacks = [model_checkpoint, lr_monitor, early_stopping]  # , viz

    logger = pl.loggers.wandb.WandbLogger(name=cfg.name, save_dir=cfg.save_dir)

    trainer = pl.Trainer(
        logger=logger,  # mlf_logger,  #
        default_root_dir=cfg.default_root_dir,
        num_nodes=cfg.num_nodes,
        num_processes=4,
        gpus=cfg.gpus,
        auto_select_gpus=True,
        overfit_batches=cfg.overfit_batches,
        track_grad_norm=cfg.track_grad_norm,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.max_epochs,
        limit_train_batches=1.0,  # FIXME:  epoch_division
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        # log_every_n_steps=50,
        progress_bar_refresh_rate=1,  # FIXME:
        accelerator=cfg.accelerator,
        sync_batchnorm=cfg.sync_batchnorm,
        precision=cfg.precision,
        weights_summary="top",
        weights_save_path=cfg.weights_save_path,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        profiler=cfg.profiler,
        benchmark=cfg.benchmark,
        deterministic=cfg.deterministic,
        # auto_lr_find=cfg.auto_lr_find,
        # auto_scale_batch_size=cfg.auto_scale_batch_size,
        plugins=cfg.plugins,  # 'ddp_sharded'
        amp_backend="native",
        amp_level="O2",
        # callbacks
        callbacks=callbacks,
    )

    if cfg.data_version == 2:
        from inputs.cxr_dm_2 import CXRDataModule
    else:
        from inputs.cxr_dm import CXRDataModule

    if cfg.auxiliary:
        from lit_model_aux import LitModel
        from inputs.cxr_dm_aux import CXRDataModule
    else:
        from lit_model import LitModel

    if cfg.classify_2class:
        from lit_model_2class import LitModel
        from inputs.cxr_dm_2class import CXRDataModule

    cxrdm = CXRDataModule(cfg)
    model = LitModel(cfg)

    trainer.fit(model, datamodule=cxrdm)
    # trainer.test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--default_root_dir", "--root_dir", type=str, default=None)
    # '/home/minki/cxr/reproduce'
    parser.add_argument(
        "--save_dir", type=str, default="/data/minki/kaggle/siim-covid19/lightning"
    )
    # parser.add_argument("--weights_save_path", "--weight_dir", type=str, default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/minki/kaggle/siim-covid19/image_1280",
    )

    # Resource
    parser.add_argument("--gpus", "--g", default="0")  # "7", "2,3"
    parser.add_argument("--num_nodes", "--nn", type=int, default=1)
    parser.add_argument("--num_workers", "--nw", type=int, default=8)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--deterministic", action="store_false")

    # Accelerator
    parser.add_argument("--precision", type=int, default=16)  # 32
    parser.add_argument("--accelerator", type=str, default=None)  # 'ddp' # Do not Use
    parser.add_argument("--sync_batchnorm", action="store_true")  # 'ddp'

    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast_dev_run", type=int, default=0)
    parser.add_argument("--track_grad_norm", type=int, default=-1)
    parser.add_argument("--overfit_batches", type=float, default=0.0)  # 0.01
    parser.add_argument("--profiler", type=str, default=None)  # 'simple'

    # Data
    parser.add_argument("--fold_index", "--fold", "--f", type=int, default=0)
    parser.add_argument("--batch_size", "--batch", type=int, default=6)
    parser.add_argument("--auto_scale_batch_size", default=None)  # 'power'
    parser.add_argument("--image_size", type=int, default=640)
    # parser.add_argument("--neg_ratio", "--neg", type=float, default=1.0)
    parser.add_argument("--label_smoothing", "--smooth", type=float, default=0.1)
    parser.add_argument("--data_version", "--dv", type=int, default=2)
    parser.add_argument("--augment_class", "--ac", action="store_true")

    # Model
    parser.add_argument("--model", type=str, default="tf_efficientnet_b7_ns")
    # tf_efficientnet_l2_ns, tf_efficientnetv2_l_in21ft1k, "tf_efficientnet_b7_ns"
    parser.add_argument("--pretrained", action="store_false")  # 'simple'
    parser.add_argument("--drop_rate", type=float, default=0.5)  # 'simple'
    parser.add_argument("--in_channels", type=int, default=3)  # 'simple'
    parser.add_argument("--auxiliary", "--aux", action="store_false")  # 'simple'
    parser.add_argument("--classify_2class", "--c2", action="store_true")  # 'simple'

    # Opts
    parser.add_argument("--auto_lr_find", action="store_true")  # Do not Use
    parser.add_argument("--lr", type=float, default=3e-6)  # 7e-5
    parser.add_argument("--loss", type=str, default="ce")  # 'simple'
    parser.add_argument("--optimizer", type=str, default="adam")  # 'simple'
    parser.add_argument("--scheduler", type=str, default="cosine")  # 'simple'
    parser.add_argument("--aux_weight", type=float, default=0.5)  # 'simple'

    # Train
    parser.add_argument("--resume_from_checkpoint", "--resume", type=str, default=None)
    parser.add_argument("--max_epochs", "--max_ep", type=int, default=100)
    parser.add_argument(
        "--stochastic_weight_avg", "--swa", action="store_true"
    )  # Do not Use
    parser.add_argument("--limit_train_batches", type=float, default=0.5)

    # Validation
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--metric", type=str, default="ap/valid")

    # Etc
    parser.add_argument("--plugins", type=str, default=None)  # "ddp_sharded"
    parser.add_argument("--logger", type=str, default=True)
    parser.add_argument("--logging_batch_interval", type=int, default=300)

    cfg = parser.parse_args()
    # cfg.weights_save_path = f"./logs/{cfg.name}"
    cfg.weights_save_path = f"{cfg.save_dir}/ckpt/{cfg.name}"

    # FIXME: Need to specify gpu for DDP usage
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
    if type(cfg.gpus) == str:
        cfg.gpus = [int(g) for g in cfg.gpus.split(",")]  # right format: [0], [2,3,4]
        cfg.gpus = ",".join([str(i) for i in range(len(cfg.gpus))])

    # DEBUG:
    if cfg.debug:
        cfg.fast_dev_run = 20
        cfg.track_grad_norm = 2
        cfg.profiler = "simple"

    pl.seed_everything(cfg.seed)

    cfg.metric_mode = "max"
    if (cfg.metric == "vacc") or cfg.metric == "vmap":
        cfg.metric_mode = "max"

    main(cfg)
