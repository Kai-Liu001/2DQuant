from functools import partial
import os
import time
import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.optim

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.dist_util import master_only
from basicsr.archs import build_network
from basicsr.losses import build_loss
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from basicsr.utils.registry import MODEL_REGISTRY

from basicsr.archs.quant_arch import QuantConv2d, QuantLinear, QuantLinearQKV, FakeQuantizerAct, FakeQuantizerBase
from basicsr.archs.swinir_arch import SwinTransformerBlock, RSTB
from tqdm import tqdm
from basicsr.utils import get_root_logger, imwrite, tensor2img
from os import path as osp
from basicsr.metrics import calculate_metric
import functools
from torch.optim import Adam
from torchvision.transforms import Resize

@MODEL_REGISTRY.register()
class TDQuantModel:
    def __init__(self, opt):
        self.opt = opt
        self.logger = get_root_logger()
        self.device = torch.device("cuda" if opt["num_gpu"] != 0 else "cpu")
        self.is_train = opt["is_train"]
        self.schedulers = []  # contains all the scheduler
        self.optimizers = []  # contains all the optim

        self.net_F = build_network(opt["network_Q"])  # FP model
        self.load_network(
            self.net_F,
            self.opt["pathFP"]["pretrain_network_FP"],
            self.opt["pathFP"]["strict_load_FP"],
            "params",
        )
        self.net_F = self.net_F.to(self.device).eval()
        
        self.net_Q = build_network(opt["network_Q"])  # FP model
        self.load_network(
            self.net_Q,
            self.opt["pathFP"]["pretrain_network_FP"],
            self.opt["pathFP"]["strict_load_FP"],
            "params",
        )


        self.build_quantized_network()  # Quantized model self.net_Q
        self.net_Q = self.net_Q.to(self.device)
        
        
        if self.opt['path']['pretrain_network_Q'] != None:
            self.load_network(
                self.net_Q,
                self.opt["path"]["pretrain_network_Q"],
                self.opt["path"]["strict_load_Q"],
                "params",
            )
            for name, module in self.net_Q.named_modules():
                if isinstance(module, FakeQuantizerBase):
                    module:FakeQuantizerBase
                    module.calibrated = True
            if self.opt.get('quant_conv', False):
                for name, module in self.net_Q.named_modules():
                    if isinstance(module, RSTB):
                        c:QuantConv2d = module.conv
                        c.act_quantizer.calibrated = True
                        c.act_quantizer.dynamic = True
                        
                        c.weight_quantizer.calibrated = False

            self.be_ready_for_calibration()
            self.calibration()
        else:
            self.be_ready_for_calibration()
            self.calibration()

        if self.is_train:
            self.init_training_settings()

    def be_ready_for_calibration(self):
        self.cali_data = torch.load(self.opt['cali_data'])

    def calibration(self):
        lq = self.cali_data['lq'].to(self.device)
        with torch.no_grad():
            _ = self.net_Q(lq)
            del _
            torch.cuda.empty_cache()
        del self.cali_data


    def init_training_settings(self) -> None:
        self.net_F.eval()
        self.net_Q.eval()

        # define losses
        train_opt = self.opt["train"]
        # |O_F-O_Q|
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        # |F_F-F_Q|
        if train_opt.get("feature_loss"):
            self.feature_loss = build_loss(train_opt["feature_loss"]).to(self.device)
        else:
            self.feature_loss = None
        

        self.setup_optimizers()
        self.setup_schedulers()
        self.build_hooks_on_Q_and_F()

    def setup_optimizers(self) -> None:
        from basicsr.archs.quant_arch import QuantLinear
        
        logger = get_root_logger()
        train_opt = self.opt["train"]

        optim_bound_params = []
        for name, module in self.net_Q.named_modules():
            if isinstance(module, QuantLinear):
                for name, param in module.named_parameters():
                    if name.endswith('bound'):
                        optim_bound_params.append(param)
                        logger.info(f'{name} is added in optim_bound_params')

        for name, module in self.quant_act.items():
            for name, param in module.named_parameters():
                if name.endswith('bound'):
                    optim_bound_params.append(param)
                    logger.info(f'{name} is added in optim_bound_params')
                        

        if train_opt.get('optim_bound', False):
            logger.info(f'optim_bound is build')
            optim_bound_type = train_opt["optim_bound"].pop("type")
            self.optimizer_bound = self.get_optimizer(
                optim_bound_type, optim_bound_params, **train_opt["optim_bound"]
            )
            self.optimizers.append(self.optimizer_bound)
        

    def build_quantized_network(self):
        self.quant_linears:dict[str, QuantLinear] = {}
        self.quant_act:dict[str, FakeQuantizerAct] = {}
        if self.opt.get('quant_conv', False):
            self.quant_conv:dict[str, QuantConv2d] = {}

        def replace_linear(linear: QuantLinear, config:dict, qkv:bool=False):
            if qkv:
                q_linear = QuantLinearQKV(config)
            else:
                q_linear = QuantLinear(config)
            q_linear.set_param(linear)
            q_linear.set_quant_flag(True)
            return q_linear
        if self.opt.get('quant_conv', False):
            def replace_conv(conv: QuantConv2d, config:dict):
                q_conv = QuantConv2d(config)
                q_conv.set_param(conv)
                q_conv.set_quant_flag(True)
                return q_conv
        
        config = {
            'bit': self.opt['bit'],
        }
        
        for name, module in self.net_Q.named_modules():
            if isinstance(module, SwinTransformerBlock):
                module.attn.qkv = replace_linear(module.attn.qkv, config, qkv=self.opt['quant']['qkv_separation'])
                module.attn.proj = replace_linear(module.attn.proj, config)
                module.mlp.fc1 = replace_linear(module.mlp.fc1, config)
                module.mlp.fc2 = replace_linear(module.mlp.fc2, config)
                
                if self.opt['quant']['qkv_separation']:
                    self.quant_linears[f"{name}.attn.qkv.q"] = module.attn.qkv.q
                    self.quant_linears[f"{name}.attn.qkv.k"] = module.attn.qkv.k
                    self.quant_linears[f"{name}.attn.qkv.v"] = module.attn.qkv.v
                else:
                    self.quant_linears[f"{name}.attn.qkv"] = module.attn.qkv
                self.quant_linears[f"{name}.attn.proj"] = module.attn.proj
                self.quant_linears[f"{name}.mlp.fc1"] = module.mlp.fc1
                self.quant_linears[f"{name}.mlp.fc2"] = module.mlp.fc2

                module.attn.mymodule_q = FakeQuantizerAct(self.opt['bit'])
                module.attn.mymodule_k = FakeQuantizerAct(self.opt['bit'])
                module.attn.mymodule_v = FakeQuantizerAct(self.opt['bit'])
                module.attn.mymodule_a = FakeQuantizerAct(self.opt['bit'])
                
                self.quant_act[f'{name}.attn.mymodule_q'] = module.attn.mymodule_q
                self.quant_act[f'{name}.attn.mymodule_k'] = module.attn.mymodule_k
                self.quant_act[f'{name}.attn.mymodule_v'] = module.attn.mymodule_v
                self.quant_act[f'{name}.attn.mymodule_a'] = module.attn.mymodule_a
            if self.opt.get('quant_conv', False):
                if isinstance(module, RSTB):
                    module.conv = replace_conv(module.conv, config)
                    self.quant_conv[f'{name}.conv'] = module.conv

        
    def build_hooks_on_Q_and_F(self):
        from basicsr.archs.swinir_arch import BasicLayer, SwinTransformerBlock

        self.feature_F = []
        self.feature_Q = []

        if self.opt["quant"]["hook_per_layer"]:
            hook_type = BasicLayer
        elif self.opt["quant"]["hook_per_block"]:
            hook_type = SwinTransformerBlock

        def hook_layer_forward(
            module: Module, input: Tensor, output: Tensor, buffer: list
        ):
            buffer.append(output)

        for name, module in self.net_F.named_modules():
            if isinstance(module, hook_type):
                module.register_forward_hook(
                    partial(hook_layer_forward, buffer=self.feature_F)
                )
        for name, module in self.net_Q.named_modules():
            if isinstance(module, hook_type):
                module.register_forward_hook(
                    partial(hook_layer_forward, buffer=self.feature_Q)
                )
        
    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize_parameters(self, current_iter):
        
        self.net_Q.eval()
        self.feature_Q.clear()
        self.feature_F.clear()

        self.output_Q = self.net_Q(self.lq)
        with torch.no_grad():
            self.output_F = self.net_F(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
                
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_Q, self.output_F) / self.output_Q.numel() * self.output_Q.size(0)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        # feature loss
        if self.feature_loss:
            l_feature = 0
            idx = 0
            for feature_q, feature_f in zip(self.feature_Q, self.feature_F):
                feature_q:torch.Tensor
                norm_q = torch.norm(feature_q, dim=(1,2)).detach()
                norm_f = torch.norm(feature_f, dim=(1,2)).detach()
                
                norm_q = norm_q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                norm_f = norm_f.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                
                feature_q = feature_q / norm_q
                feature_f = feature_f / norm_f
                
                fi = self.feature_loss(feature_q, feature_f) / feature_q.numel()
                
                loss_dict[f"l_feature_{idx}"] = fi
                l_feature += fi
                idx += 1

            loss_dict["l_feature"] = l_feature
            l_total += l_feature

        self.optimizer_bound.zero_grad()
        l_total.backward()
        self.optimizer_bound.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        self.save_network(self.net_Q, "net_Q", current_iter)
        self.save_training_state(epoch, current_iter)

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt["dist"]:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")
            
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]])
                metric_data["img2"] = gt_img
                del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    if name == 'psnr':
                        psnr = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    
            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.png",
                    )
                else:
                    if self.opt["val"].get("suffix", False):
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{psnr:.4f}_.png',
                        )
                imwrite(sr_img, save_img_path)
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        # self.time_forward = 0.0
        # self.time_get_visual = 0.0
        # self.time_save_img = 0.0
        # self.time_cal_metric = 0.0
        # self.time_total = 0.0
        
        # ,time_forward:{self.time_forward:.2f},time_get_visual:{self.time_get_visual:.2f}')
        if use_pbar:
            pbar.close()
        

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        # print(f'time_forward:{self.time_forward:.2f}')
        # print(f'time_get_visual:{self.time_get_visual:.2f}')
        # print(f'time_save_img:{self.time_save_img:.2f}')
        # print(f'time_cal_metric:{self.time_cal_metric:.2f}')
        # print(f'time_total:{self.time_total:.2f}')


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def test(self):
        # pad to multiplication of window_size
        # we do not use self-ensamble.
        if not self.opt['quant'].get('self_ensamble', False):
            
            window_size = self.opt["network_Q"]["window_size"]
            scale = self.opt.get("scale", 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            # 修改（按源码）
            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w + mod_pad_w]
            
            if self.opt['quant'].get('bicubic', False):
                # print(img.size())
                # exit(0)
                resize = Resize((img.size(2) * self.opt['network_Q']['upscale'], img.size(3) * self.opt['network_Q']['upscale']))
                self.output = resize(img)
            elif self.opt['quant'].get('fp_test', False):
                self.net_F.eval()
                with torch.no_grad():
                    self.output = self.net_F(img)
            else:
                if hasattr(self, "net_g_ema"):
                    self.net_g_ema.eval()
                    with torch.no_grad():
                        self.output = self.net_g_ema(img)
                else:
                    self.net_Q.eval()
                    with torch.no_grad():
                        self.output = self.net_Q(img)
                    self.net_Q.eval()

            _, _, h, w = self.output.size()
            self.output = self.output[
                :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
            ]
        else:
            transforms = [
                (lambda x: x, lambda y: y),  # No-op
                (lambda x: x.flip(2), lambda y: y.flip(2)),  # Vertical flip
                (lambda x: x.flip(3), lambda y: y.flip(3)),  # Horizontal flip
                (lambda x: x.flip(2).flip(3), lambda y: y.flip(2).flip(3)),  # Vertical + Horizontal flip
                (lambda x: x.transpose(2, 3), lambda y: y.transpose(2, 3)),  # Rotate 90 degrees
                (lambda x: x.transpose(2, 3).flip(2), lambda y: y.flip(2).transpose(2, 3)),  # Rotate 90 degrees + Vertical flip
                (lambda x: x.transpose(2, 3).flip(3), lambda y: y.flip(3).transpose(2, 3)),  # Rotate 90 degrees + Horizontal flip
                (lambda x: x.transpose(2, 3).flip(2).flip(3), lambda y: y.flip(2).flip(3).transpose(2, 3))  # Rotate 90 degrees + Vertical + Horizontal flip
            ]
            window_size = self.opt["network_Q"]["window_size"]
            scale = self.opt.get("scale", 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            # 修改（按源码）
            # real_outputs = []
            inputs = []
            
            for transform, inverse_transform in transforms:
                img = self.lq
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + mod_pad_h, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w + mod_pad_w]
                inputs.append(transform(img))
                # print(img.size())
            batch_input_1 = torch.cat(inputs[:4], dim=0)
            batch_input_2 = torch.cat(inputs[4:], dim=0)
            
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                with torch.no_grad():
                    output1 = self.net_g_ema(batch_input_1)
            else:
                self.net_Q.eval()
                with torch.no_grad():
                    output1 = self.net_Q(batch_input_1)
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                with torch.no_grad():
                    output2 = self.net_g_ema(batch_input_2)
            else:
                self.net_Q.eval()
                with torch.no_grad():
                    output2 = self.net_Q(batch_input_2)
                # real_outputs.append(output)
            # output = output1 + output2
            # output = torch.cat([output1, output2], dim=0)
            outputs1 = torch.chunk(output1, len(transforms), dim=0)
            outputs2 = torch.chunk(output2, len(transforms), dim=0)
            outputs = outputs1 + outputs2
            inverse_transformed_outputs = []
            for output, (_, inv_transform) in zip(outputs, transforms):
                inv_output = inv_transform(output)
                _, _, h_out, w_out = inv_output.size()
                cropped_output = inv_output[:, :, 0: h_out - mod_pad_h * scale, 0: w_out - mod_pad_w * scale]
                inverse_transformed_outputs.append(cropped_output)
                
            # Compute the mean of all the processed outputs
            final_output = torch.mean(torch.stack(inverse_transformed_outputs), dim=0)
            self.output = final_output
            return final_output

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if (
            hasattr(self, "best_metric_results")
            and dataset_name in self.best_metric_results
        ):
            return
        elif not hasattr(self, "best_metric_results"):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt["val"]["metrics"].items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay
            )

    def get_current_log(self):
        bits_weight = []
        bits_act = []
        size_total_weight = 0
        size_total_act = 0
        for name, module in self.net_Q.named_modules():
            if isinstance(module, QuantLinear):
                self.log_dict[f"param_{name}_act_lb"] = float(
                    module.act_quantizer.lower_bound
                )
                self.log_dict[f"param_{name}_act_ub"] = float(
                    module.act_quantizer.upper_bound
                )
                self.log_dict[f"param_{name}_act_n_bit"] = float(
                    module.act_quantizer.n_bit
                )

                self.log_dict[f"param_{name}_weight_lb"] = float(
                    module.weight_quantizer.lower_bound
                )
                self.log_dict[f"param_{name}_weight_ub"] = float(
                    module.weight_quantizer.upper_bound
                )
                self.log_dict[f"param_{name}_weight_n_bit"] = float(
                    module.weight_quantizer.n_bit
                )
                size_total_weight += module.weight_quantizer.size_of_input
                size_total_act += module.act_quantizer.size_of_input
                
                bits_weight.append(float(module.weight_quantizer.n_bit * module.weight_quantizer.size_of_input))
                bits_act.append(float(module.act_quantizer.n_bit * module.act_quantizer.size_of_input))
                
                
        self.log_dict['param_weight_avg_n_bit'] = np.sum(bits_weight) / size_total_weight
        self.log_dict['param_act_avg_n_bit'] = np.sum(bits_act) / size_total_act

        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif self.opt["num_gpu"] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs) -> Adam:
        if optim_type == "Adam":
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt["train"]
        scheduler_type = train_opt["scheduler"].pop("type")
        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.MultiStepLR(optimizer,**train_opt["scheduler"])
                )
                
        
        elif scheduler_type == "CosineAnnealingRestartLR":
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt["scheduler"]
                    )
                )
        
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                s = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.opt['train']['total_iter']
                    )
            
                self.schedulers.append(s)
        elif scheduler_type == '':
            for optimizer in self.optimizers:
                s = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.opt['train']['total_iter']
                    )
            
                self.schedulers.append(s)
        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_type} is not implemented yet."
            )

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key="params"):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = "latest"
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = os.path.join(self.opt["path"]["models"], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key
        ), "The lengths of net and param_key should be the same."

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith("module."):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(
                    f"Save model error: {e}, remaining retry times: {retry - 1}"
                )
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f"Still cannot save {save_path}. Just ignore it.")
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning("Current net - loaded net:")
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f"  {v}")
            logger.warning("Loaded net - current net:")
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f"  {v}")

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f"Size different, ignore [{k}]: crt_net: "
                        f"{crt_net[k].shape}; load_net: {load_net[k].shape}"
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key="params"):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
                logger.info("Loading: params_ema does not exist, use params.")
            load_net = load_net[param_key]
        logger.info(
            f"Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}]."
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
            }
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.opt["path"]["training_states"], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(
                        f"Save training state error: {e}, remaining retry times: {retry - 1}"
                    )
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f"Still cannot save {save_path}. Just ignore it.")
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt["dist"]:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt["rank"] == 0:
                    losses /= self.opt["world_size"]
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
