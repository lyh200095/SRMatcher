from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from model.loftr_src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from model.loftr_src.losses.loftr_loss import LoFTRLoss, GeoLoss
from model.loftr_src.optimizers import build_optimizer, build_scheduler
from model.loftr_src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from model.loftr_src.utils.plotting import make_matching_figures
from model.loftr_src.utils.comm import gather, all_gather
from model.loftr_src.utils.misc import lower_config, flattenList
from model.loftr_src.utils.profiler import PassThroughProfiler
from model.full_model import GeoFormer
import os
import torch.distributed as dist

class PL_SRMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
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

        # manage required_grad
        import re
        if self.config.TRAINER.FREEZE_DINO and True and self.matcher.dinov2 is not None:
            omit = {'patch_embed.proj.weight', 'patch_embed.proj.bias', 'pos_embed'}
            omit = {}
            print('freeze dino')

            dinov2_block_att_paras = {
                "blocks.{}{}".format(layer, pattern)
                for pattern in ['.attn.qkv.', '.norm1.'] for layer in []
                # for pattern in ['.attn.qkv.', '.norm1.'] for layer in range(12)
                # for pattern in ['.attn.qkv.', ] for layer in [4,5,10,11]
                # for pattern in ['.attn.qkv.', ] for layer in [5, 11]
                # for pattern in ['.attn.qkv.', '.norm1.'] for layer in [9, 10, 11]
                # for pattern in ['.attn.qkv.', '.norm1.'] for layer in [11]
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

        self.test_outputs = []
        self.val_outputs = defaultdict(list)
        self.train_outputs = []

        from homodataset.hpatches import Hpatches_Eval
        self.hpatches = Hpatches_Eval(self.matcher, self.loftr_cfg, plmodel=self)

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    # def optimizer_step(
    #         self, epoch, batch_idx, optimizer, optimizer_idx,
    #         optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP

        # 修改 global_step 定义，改为分阶段
        # global_step = self.trainer.global_step
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

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

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
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data,
                    self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)

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
        if dataloader_idx == 1:
            ret = self.validation_step_hpatches(batch, batch_idx)
            self.val_outputs[1].append(ret)
            return
        else:
            pass
            # return
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        ret = {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        self.val_outputs[0].append(ret)
        return None

    def on_validation_epoch_end(self, ):
        from copy import deepcopy
        self.on_validation_epoch_end_hpatches(self.val_outputs[1])
        # outputs = self.val_outputs[0]
        outputs = deepcopy(self.val_outputs[0])

        self.val_outputs = defaultdict(list)
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            # if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
            #     cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def validation_step_hpatches(self, batch, batch_idx, ):

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

        return ret

    def on_validation_epoch_end_hpatches(self, eval_outputs):
        # hpatch
        outputs_gather = self.all_gather(eval_outputs)  # 需要保证 out_dict 为字段一致的List[Dict[Tensor]]]
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
            # print(out_dict['index'])
            for k, v in zip(['1px', '3px', '5px', '10px', ], auc):
                auc_dict['auc@' + k] = v

            if self.trainer.is_global_zero and total_val_samples >= 16:  # and self.global_step > 0:
                text = 'epoch={}\n'.format(self.current_epoch)
                print('[r{}] hpatch auc (len={}): {}'.format(self.global_rank, total_val_samples, auc))

        self.log('val_auc@3px', auc[1], on_step=False, on_epoch=True)
        self.log('val/auc@3px', auc[1], on_step=False, on_epoch=True)
        self.log('val/auc@5px', auc[2], on_step=False, on_epoch=True)
        self.log('val/auc@10px', auc[3], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("GeoFormer"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
        self.test_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self, ):
        # metrics: dict of list, numpy
        outputs = self.test_outputs
        _metrics = [o['metrics'] for o in outputs]
        inliers = flattenList(x['inliers'] for x in _metrics)
        # inliers2 = self.all_gather(inliers[0])
        inliers
        # metrics = {k: flattenList(self.all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
        # metrics = gather(_metrics)
        # ['identifiers', 'epi_errs', 'R_errs', 't_errs', 'inliers']

        # Initialize an empty dict to store gathered metrics
        gathered_metrics = {k: [] for k in _metrics[0].keys()}

        import torch
        import torch.distributed as dist

        def gather_records(records, device=None):
            maxlen = 0
            for _ in records:
                _['len'] = len(_['inliers'][0])
                maxlen = max(maxlen, _['len'])

            for _ in records:
                for k in ['inliers']:
                    # pad
                    pass

                _.pop('epi_errs')
                _.pop('inliers')

            return records

        # dist.destroy_process_group()
        # metrics = gather_records(_metrics)
        # metrics2 = self.all_gather(metrics[0]['identifiers'])


        FILE_NAME = "./_tmp_process_data_rank_{}.pt"
        # 假设这是在某个PyTorch Lightning模块的方法中
        def save_process_data(self, data):
            # 获取当前进程的全局rank
            global_rank = self.trainer.global_rank
            # 为每个进程生成一个唯一的文件名
            filename = FILE_NAME.format(global_rank)
            # 使用torch.save保存数据
            torch.save(data, filename)

        def load_and_merge_data(self, num_processes):
            # 确保所有进程都已完成数据保存
            # torch.distributed.barrier()

            # 只有主进程执行读取和合并操作
            if self.trainer.is_global_zero:
                all_data = []
                for rank in range(num_processes):
                    filename = FILE_NAME.format(rank)
                    process_data = torch.load(filename)
                    all_data.extend(process_data)
                    # 删除文件（可选）
                    os.remove(filename)
                # 这里的all_data包含了所有进程的数据

                data = defaultdict(list)
                for _ in all_data:
                    for k, v in _.items():
                        if isinstance(v, list) and len(v) == 1:
                            v = v[0]
                        data[k].append(v)
                return data

        save_process_data(self, _metrics)
        torch.distributed.barrier()

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        metrics = load_and_merge_data(self, torch.distributed.get_world_size())
        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)

        # torch.distributed.barrier()
        # self.trainer.strategy.barrier()

