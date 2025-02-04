"""Trains the face-shifter network.

Example train usage:
python aei_trainer.py -c config/p4d.24xlarge.yaml -g 0,1,2,3,4,5,6,7 -n vggface2only --checkpoint_path /SHARED/epoch1.ckpt

Example eval usage:
python aei_trainer.py -c config/p4d.24xlarge.yaml -g 0,1,2,3,4,5,6,7 -n vggface2only --eval_only --checkpoint_path /SHARED/epoch1.ckpt
"""

import os
from omegaconf import OmegaConf
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from aei_net import AEINet


def main(args):
    hp = OmegaConf.load(args.config)
    model = AEINet(hp)
    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    val_saver = ModelCheckpoint(
        dirpath=hp.log.chkpt_dir,
        filename='{epoch}_{val_loss:.4f}' + args.name,
        monitor='val_loss',
        verbose=True,
        save_top_k=args.save_top_k,
    )
    
    periodic_saver = ModelCheckpoint(
        dirpath=hp.log.chkpt_dir,
        filename='{epoch}_{step}' + args.name,
        every_n_train_steps=10000,
        save_top_k=-1, # prevents overwriting
        verbose=True,
    )

    # should on_train_end be set?
    trainer = Trainer(
        logger=pl_loggers.TensorBoardLogger(hp.log.log_dir),
        callbacks=[val_saver, periodic_saver],
        weights_save_path=save_path,
        gpus=-1 if args.gpus is None else args.gpus,
        distributed_backend='ddp',
        num_sanity_val_steps=1,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=hp.model.grad_clip,
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_interval,
        progress_bar_refresh_rate=1,
        max_epochs=100,
        accumulate_grad_batches=args.accumulate_grad_batches,
        stochastic_weight_avg=args.stochastic_weight_avg,
    )
    if args.eval_only:
        trainer.validate(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-g', '--gpus', type=str, default=None,
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")
    parser.add_argument('-s', '--save_top_k', type=int, default=-1,
                        help="save top k checkpoints, default(-1): save all")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument('--val_interval', type=float, default=0.1,
                        help="https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#val-check-interval")
    parser.add_argument('--eval_only', action='store_true',
                        help="...")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help="...")
    parser.add_argument('--stochastic_weight_avg', action='store_true',
                        help="...")

    args = parser.parse_args()

    main(args)
