"""
 Some functions in this file is copied from https://github.com/yhhhli/BRECQ
 and modified for this project's needs.
"""
import math
import random
from typing import Sequence, Union

import torch
from tqdm import tqdm

from compression.module.compression_wrapper import CompressionWrapper
from compression.quantization.adaround_utils import get_soft_targets


## TODO: remove before publish
import wandb


class CosineTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            t1, t2 = t - self.start_decay, self.t_max - self.start_decay
            return self.end_b + (self.start_b - self.end_b) * (1 + math.cos(math.pi * t1 / t2)) / 2


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    def __init__(self,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 reg_factor=0.3,
                 n_iters=100):

        self.n_iters = n_iters
        self.reg_factor = reg_factor
        self.loss_start = n_iters * warmup
        self.p = p
        self.count = 0

        b_decay_mode = 'linear'
        if b_decay_mode == 'linear':
            temp_decay_object = LinearTempDecay
        elif b_decay_mode == 'cosine':
            temp_decay_object = CosineTempDecay
        else:
            raise NotImplemented
        self.temp_decay = temp_decay_object(n_iters, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                            start_b=b_range[0], end_b=b_range[1])

    def reset(self):
        self.count = 0

    def __call__(self,
                 pred: torch.Tensor,
                 tgt: torch.Tensor,
                 soft_targets: list[torch.Tensor],
                 bitwidths: Union[int, Sequence[int], torch.Tensor]):
        """
        Parameters
        ----------
        pred, tgt     : layer outputs (quantised vs float).
        soft_targets  : list of soft gates `s` (same shapes as weights).
        bitwidths     : int or 1-D list / tensor of per-channel bit-widths.
                        Length must equal out-channels of every tensor in
                        `soft_targets`.
        """
        # ------------------------------------------------------------------
        # 0. reconstruction loss
        rec_loss = torch.mean(torch.abs(pred - tgt).pow(self.p))

        # current temperature b
        b = (self.temp_decay.start_b if self.count < self.loss_start
             else self.temp_decay(self.count))

        # prepare bit-width tensor (broadcast later)
        if isinstance(bitwidths, (int, float)):
            bw_tensor = None          # scalar path
            is_binary_scalar = (bitwidths == 1)
        else:
            bw_tensor = torch.as_tensor(bitwidths, device=soft_targets[0].device)
            if bw_tensor.ndim != 1:
                raise ValueError("bitwidths list / tensor must be 1-D")

        # ------------------------------------------------------------------
        # 1. rounding penalty per tensor
        round_loss = 0.0
        for st in soft_targets:                                  # iterate gated params
            if bw_tensor is None:                                # scalar bit-width
                if is_binary_scalar:
                    penalty = 1 - torch.abs(2 * st - 1).pow(b)   # binary
                else:
                    penalty = 1 - torch.abs((st - .5) * 2).pow(b)
            else:
                if st.shape[0] != bw_tensor.numel():
                    raise ValueError("Mismatch between bitwidths length and "
                                     "first dimension of soft-target tensor")

                # mask broadcasted to tensor shape
                mask = (bw_tensor == 1).view(-1, *([1] * (st.ndim - 1)))

                penalty_bin  = 1 - torch.abs(2 * st - 1).pow(b)
                penalty_mult = 1 - torch.abs((st - .5) * 2).pow(b)
                penalty      = torch.where(mask, penalty_bin, penalty_mult)

            round_loss += self.reg_factor * penalty.mean()

        # ------------------------------------------------------------------
        total_loss = rec_loss + round_loss
        self.count += 1

        if (self.count % max(1, self.n_iters // 10) == 0
                or self.count == self.temp_decay.t_max):
            print(f'Total loss:\t{total_loss:.3f}  '
                  f'(rec:{rec_loss:.3f}, round:{round_loss:.3f})  '
                  f'b={b:.2f}\tstep={self.count}')

        return total_loss, {"rec_loss": rec_loss,
                            "round_loss": round_loss / self.reg_factor,
                            "b": b}

    # def __call__(self, pred, tgt, soft_targets, bitwidths):
    #     rec_loss = torch.mean(torch.pow(torch.abs(pred - tgt), self.p))
    #     b = self.temp_decay.start_b if self.count < self.loss_start else self.temp_decay(self.count)
    #     round_loss = sum([self.reg_factor * (1 - ((st - .5).abs() * 2).pow(b)).mean() for st in soft_targets])
    #
    #     total_loss = rec_loss + round_loss
    #
    #     self.count += 1
    #
    #     if self.count % max(1, int(self.n_iters / 10)) == 0 or self.count == self.temp_decay.t_max:
    #         print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
    #             float(total_loss), float(rec_loss), float(round_loss), b, self.count))
    #     return total_loss, {"rec_loss": rec_loss, "round_loss": float(round_loss) / self.reg_factor, "b": b}

def model_register_hook(layer, layer2tensor, collect_input=True, collect_output=True):
    def hook(model, input, output):
        layer2tensor.append(([i.detach().clone() for i in input] if collect_input else None,
                             output.detach().clone() if collect_output else None))

    return layer.register_forward_hook(hook)


class FineTuning:

    def __init__(self, representative_dataset, model_manager, iters=2000, lr=0.01, reg_factor=0.3, batch_size=32, wandb_en=False):
        self.samples = torch.cat([b for b, _ in representative_dataset], dim=0)
        self.samples = model_manager.data_to_device(self.samples)
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.iters = iters
        self.lr = lr
        self.reg_factor = reg_factor

        ## TODO: remove before publish
        self.wandb_en = wandb_en

    def __call__(self, compressed_model, float_model):
        # Set Loss function
        loss_func = LossFunction(b_range=(20, 2), decay_start=0, warmup=0.2,
                                 reg_factor=self.reg_factor, n_iters=self.iters)

        for n, q_layer in compressed_model.named_modules():
            if isinstance(q_layer, CompressionWrapper):
                loss_func.reset()

                q_layer.init_layer_reconstruction()

                # fine-tune
                f_layer = [mm for nn, mm in float_model.named_modules() if nn == n][0]
                f_activation_tensor = []
                f_handle = model_register_hook(f_layer, f_activation_tensor, collect_input=False)
                q_activation_tensor = []
                q_handle = model_register_hook(q_layer, q_activation_tensor, collect_output=False)
                with torch.no_grad():
                    if isinstance(self.samples, list):
                        for data in tqdm(self.samples, desc="Extract inputs & outputs"):
                            self.model_manager.forward(float_model, data)
                            self.model_manager.forward(compressed_model, data)
                    elif isinstance(self.samples, dict):
                        num_samples = self.samples['labels'].shape[0]
                        for i in tqdm(range(int(num_samples / self.batch_size)), desc="Extract inputs & outputs"):
                            data = {k: v[i * self.batch_size:(i + 1) * self.batch_size]
                                    for k, v in self.samples.items()}
                            self.model_manager.forward(float_model, data)
                            self.model_manager.forward(compressed_model, data)
                    else:
                        for data in tqdm(self.samples.split(self.batch_size), desc="Extract inputs & outputs"):
                            self.model_manager.forward(float_model, data)
                            self.model_manager.forward(compressed_model, data)

                # remove handle:
                f_handle.remove()
                q_handle.remove()
                self.apply_adaround(f_activation_tensor, loss_func, n, q_activation_tensor,
                                    q_layer)

    def apply_adaround(self, f_activation_tensor, loss_func, n, q_activation_tensor, q_layer):
        q_layer.train()
        # Set Optimizer
        weight_params, bias_params, scale_params = q_layer.get_trainable_params()
        optimizer = torch.optim.RAdam(weight_params, lr=self.lr)
        activation_tensors = [(qt[0], ft[1]) for ft, qt in zip(f_activation_tensor, q_activation_tensor)]
        # Run optimization loop
        for _ in tqdm(range(self.iters), desc=f'fine-tuning layer {n}'):
            inputs, out_full = random.sample(activation_tensors, 1)[0]
            optimizer.zero_grad()
            out_quant = q_layer(*[i.detach() for i in inputs])
            err, extra_loss_params = loss_func(out_quant, out_full.detach(),
                                               [get_soft_targets(t, q_layer.gamma, q_layer.zeta, q_layer.base_n_bits)
                                                for t in weight_params], q_layer.base_n_bits)

            ## TODO: remove before publish
            if self.wandb_en and ((_ + 1) % 1000 == 0):
                log_dict = {'loss': err, **extra_loss_params}
                wandb.log(log_dict)

            if err < 1e-8:
                break

            err.backward()
            optimizer.step()
        q_layer.eval()
        del f_activation_tensor
        del q_activation_tensor
        del activation_tensors
        torch.cuda.empty_cache()
