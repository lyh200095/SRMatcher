from collections import defaultdict

import pytorch_lightning as pl
import torch
from loguru import logger

from model.full_model import GeoFormer
from model.loftr_src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from model.loftr_src.losses.loftr_loss import LoFTRLoss, GeoLoss
from model.loftr_src.optimizers import build_optimizer, build_scheduler
from model.loftr_src.utils.metrics import (
    compute_symmetrical_epipolar_errors
)
from model.loftr_src.utils.misc import lower_config
from model.loftr_src.utils.plotting import make_matching_figures
from model.loftr_src.utils.profiler import PassThroughProfiler

import torch.distributed as dist


import logging, os

# 设置全局logger
def setup_global_logger(name, folder='logs/'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 或者根据你的需要设置其他级别
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台日志输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件日志输出
    fh = logging.FileHandler(os.path.join(folder, name + '.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# 实际设置logger
# global_logger = setup_global_logger('my_debug_logger')

class PL_H_SRMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, log_dir=None):

        super().__init__()
        # Misc
        self.config = config  # full loftr_config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = GeoFormer(loftr_config=_config['loftr'])
        self.loss = GeoLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
            del state_dict

        # manage required_grad
        import re
        if self.config.TRAINER.FREEZE_DINO and True and self.matcher.dinov2 is not None:
            omit = {'patch_embed.proj.weight', 'patch_embed.proj.bias', 'pos_embed'}
            omit = {}

            dinov2_block_att_paras = {
                "blocks.{}{}".format(layer, pattern)
                for pattern in ['.attn.qkv.', '.norm1.'] for layer in []
            }
            for n, p in self.matcher.dinov2.encoder.named_parameters():
                if n in omit:
                    p.requires_grad = True
                    if self.global_rank == 0:
                        print('retrain {} in dinov2'.format(n))
                elif True and any(n.startswith(patt) for patt in dinov2_block_att_paras):
                    # 解冻后几层的qkv
                    p.requires_grad = True
                    if self.global_rank == 0:
                        print('retrain {} in dino trans blocks'.format(n))
                else:
                    p.requires_grad = False

        
        # Testing
        self.dump_dir = dump_dir
        self.train_outputs = []
        from homodataset.hpatches import Hpatches_Eval
        self.hpatches = Hpatches_Eval(self.matcher, self.loftr_cfg, plmodel=self)
        self.eval_outputs = defaultdict(list)

        # log
        self.tlogger = setup_global_logger('train', log_dir)

    def remove_requires_grad_false(self, state_dict):
        new_state_dict = {}
        for name, param in state_dict.items():
            if 'matcher.dinov2' in name and not param.requires_grad:
                continue
            new_state_dict[name] = param
        return new_state_dict

    def save_checkpoint(self, filepath, **kwargs):
        if 'state_dict' in kwargs:
            state_dict = kwargs['state_dict']
            kwargs['state_dict'] = self.remove_requires_grad_false(state_dict)
        super(PL_H_SRMatcher, self).save_checkpoint(filepath, **kwargs)

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        # optimizer = build_optimizer(self, self.config)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.TRAINER.TRUE_LR, weight_decay=self.config.TRAINER.ADAMW_DECAY)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP

        # 修改 global_step 定义，改为分阶段
        step_per_epoch = len(self.trainer.datamodule.train_dataset) // self.config.TRAINER.WORLD_SIZE
        steps_one_stage = step_per_epoch * 20
        global_step = self.trainer.global_step % int(steps_one_stage)
        global_step = steps_one_stage - global_step

        if global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                self.logger.experiment.add_scalar(f'lr/lr', lr, self.global_step)
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("GeoFormer"):
            self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)

        # logging print info
        if self.trainer.global_rank == 0 and self.global_step < 3:
            if self.matcher.dinov2 is not None:
                if self.matcher.dinov2.layer_feat is not None:
                    print('Using dinov2.layer_feat')
                else:
                    print('Do not using dinov2.layer_feat')

        torch.cuda.empty_cache()
        ret = {'loss': batch['loss']}
        self.train_outputs.append(ret)
        return ret

    def on_train_epoch_end(self):
        outputs = self.train_outputs
        # gathered_outputs = self.all_gather(outputs)
        if outputs:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

            # 收集当前进程的平均损失
            avg_loss_local = torch.stack([x['loss'] for x in outputs]).mean()

            # 使用all_reduce来聚合所有进程的损失
            dist.all_reduce(avg_loss_local, op=dist.ReduceOp.SUM)

            # 计算全局平均损失
            avg_loss_global = avg_loss_local / dist.get_world_size()

            if self.trainer.is_global_zero:
                self.logger.experiment.add_scalar(
                    'train/avg_loss_on_epoch', avg_loss_global,
                    global_step=self.current_epoch)
   
        self.train_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        # self.log('val_loss', batch['loss'])
        if dataloader_idx == 1:
            self._trainval_inference(batch)
            self.eval_outputs[1].append({'loss': batch['loss']})
        else:
            ret = dict(corner_dist=0, irat=0, cid=0, index=torch.tensor(-1))
            if 'image0' not in batch:
                pass
            else:
                try:
                    for key in ['image0', 'image1', 'mask1', 'mask2', 'H_0to1', 'H_1to0']:
                        if key not in batch: continue
                        batch[key] = batch[key].squeeze(0)
                    for key in ['sc1', 'sc2',]:
                        batch[key] = torch.tensor(batch[key]).cpu().tolist()
                    batch['H_gt'] = batch['H_gt'].squeeze(0).cpu().numpy()
                    batch['name'] = batch['name'][0]
                    self._trainval_inference(batch)
                    ret_ = self.hpatches.cal_one(batch_idx=batch_idx, isprint=False, **batch)
                    ret.update(ret_)
                    batch.update({'ret_dict': ret_})
                    ret.update({'index': batch.get('index', torch.tensor(-1))})

                except:
                    import traceback
                    traceback.print_exc()

            self.eval_outputs[0].append(ret)

    def on_validation_epoch_end(self):
        eval_outputs = self.eval_outputs
        self.eval_outputs = defaultdict(list)

        # 计算loss
        if len(eval_outputs[1]):
            avg_loss_local = torch.stack([x['loss'] for x in eval_outputs[1]]).mean()
            dist.all_reduce(avg_loss_local, op=dist.ReduceOp.SUM)
            avg_loss_global = avg_loss_local / dist.get_world_size()
        else:
            avg_loss_global = 0
        self.log('val/loss', avg_loss_global, on_step=False, on_epoch=True)
        self.log('val_loss', avg_loss_global, on_step=False, on_epoch=True)

        # hpatch
        outputs_gather = self.all_gather(eval_outputs[0])  # 需要保证 out_dict 为字段一致的List[Dict[Tensor]]]
        auc_dict = defaultdict(float)
        if self.trainer.is_global_zero or 1:
            # 对汇总后的结果进行进一步处理
            tensor_key = [k for k, v in outputs_gather[0].items() if torch.is_tensor(v)]
            out_dict = {
                key: torch.concat([_[key] for _ in outputs_gather], dim=0)
                for key in tensor_key
            }
            out_dict['index'] = out_dict['index'].flatten()
            index_valid = (out_dict['index'] >= 0)
            out_dict = {k: v[index_valid] for k, v in out_dict.items()}
            total_val_samples = len(out_dict['index'])

            hpatches_res = self.hpatches.cal_scores(out_dict)
            auc = hpatches_res['auc_sa']
            for k, v in zip(['1px', '3px', '5px', '10px', ], auc):
                auc_dict['auc@' + k] = v

            if self.trainer.is_global_zero and total_val_samples >= 16:  # and self.global_step > 0:
                text = 'epoch={}\n'.format(self.current_epoch)
                self.tlogger.info(text)
                if getattr(self.matcher, 'kmnn1', None):
                    self.tlogger.info("kmnn1, kmnn2: {}, {}".format(self.matcher.kmnn1, self.matcher.kmnn2))
                self.tlogger.info(hpatches_res['summary'])

                print('[r{}] hpatch auc (len={}): {}'.format(self.global_rank, total_val_samples, auc))
             
        self.log('val_auc@3px', auc[1], on_step=False, on_epoch=True)
        self.log('val/auc@3px', auc[1], on_step=False, on_epoch=True)
        self.log('val/auc@5px', auc[2], on_step=False, on_epoch=True)
        self.log('val/auc@10px', auc[3], on_step=False, on_epoch=True)




