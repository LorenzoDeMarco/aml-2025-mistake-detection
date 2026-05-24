# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import shutil

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    # pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['substep1_2_output_folder']):
        os.mkdir(cfg['substep1_2_output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S")  
        ckpt_folder = os.path.join(
            cfg['substep1_2_output_folder'], cfg_filename + '_' + ts_str)
    else:
        ckpt_folder = os.path.join(
            cfg['substep1_2_output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)


    if cfg.get('use_auto_lr_scaling', False):
        total_batch_size = cfg['loader']['batch_size'] * len(cfg['devices'])
        ref_batch_size = cfg.get('ref_batch_size', 16)
        scale = total_batch_size / ref_batch_size
        cfg['opt']["learning_rate"] *= scale
        print(f"Auto-scaling learning rate by {scale:.3f} -> {cfg['opt']['learning_rate']:.2e}")

    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    # -------- create validation dataset and loader ----------
    val_dataset = None
    val_loader = None
    if cfg.get('val_split') and len(cfg['val_split']) > 0:
        val_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
        )
        val_loader = make_data_loader(
            val_dataset, False, None, 1, cfg['loader']['num_workers']
        )

    """3. create model, optimizer, and scheduler"""
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training / validation loop with early stopping and best model saving"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    best_map = -1.0
    patience = cfg['opt'].get('early_stop_patience', 10)
    no_improve_epochs = 0
    best_ckpt_path = os.path.join(ckpt_folder, 'best_model.pth.tar')

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # -------- evaluation ----------
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            # use the official evaluation code for validation
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds']
            )
            with torch.no_grad():
                mAP = valid_one_epoch(
                    val_loader,
                    model_ema.module,   
                    epoch,
                    evaluator=det_eval,
                    output_file=None,
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    print_freq=args.print_freq
                )

            if mAP > best_map:
                best_map = mAP
                no_improve_epochs = 0
                # save best checkpoint
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'state_dict_ema': model_ema.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_map': best_map,
                }
                save_checkpoint(save_states, False, file_folder=ckpt_folder, file_name='best_model.pth.tar')
                print(f"Best model saved (mAP={best_map:.4f})")
            else:
                no_improve_epochs += 1
                print(f"Validation mAP={mAP:.4f} (best={best_map:.4f}), no improvement for {no_improve_epochs} epochs")
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # save checkpoint
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict_ema': model_ema.module.state_dict(),
            }
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--val-freq', default=1, type=int,
                        help='validation frequency (default: every epoch)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)