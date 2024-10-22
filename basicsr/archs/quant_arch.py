from typing import Any
from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import torch
from torch import Tensor, FloatTensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
from functools import partial
import torch.nn.functional as F

calibrated_num = 0
total_num = 0

def quant(input:torch.Tensor, lb:float, ub:float, bit:int):
    input = input.clamp(lb, ub)
    s = (ub - lb) / (2 ** bit -1)
    input = (input - lb)/s
    input = input.round()
    input = input * s + lb
    return input

def cal_mse(input:torch.Tensor, lb:float, ub:float, bit:int):
    quant_input = quant(input, lb, ub, bit)
    res = float(torch.norm(input - quant_input))
    return res

def DOBI(input:torch.Tensor, bit:int, one_direction = False, num:int=100):
    min_value = torch.min(input)
    max_value = torch.max(input)
    
    diff = (max_value - min_value) / (2 * num)
    
    history_min = float('inf')
    input = input.cuda()
    
    if one_direction:
        diff = (max_value - min_value) / num
        for i in range(num):
            lb = min_value
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    else:
        diff = (max_value - min_value) / (2 * num)
        for i in range(num):
            lb = min_value + diff * i
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    global calibrated_num
    global total_num
    
    calibrated_num += 1
    print(f'calibration:{calibrated_num}/{total_num}')
    
    return float(best_lb), float(best_ub)

class Differentiable_Round(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs):
        return grad_outputs


class Differentiable_Clip(Function):
    @staticmethod
    def forward(
        ctx: _ContextMethodMixin,
        input: Tensor,
        min_val: Tensor,
        max_val: Tensor,
    ) -> Any:
        ctx.save_for_backward(input, min_val, max_val)
        # if isinstance(min_val, Tensor):
        #     min_val = min_val.item()
        # if isinstance(max_val, Tensor):
        #     max_val = max_val.item()
        
        # output = input.clamp(min_val, max_val)
        output = input.clamp(min_val.item(), max_val.item())
        return output

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensors

        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0
        
        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        grad_min = grad_min.sum().view(1)

        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        grad_max = grad_max.sum().view(1)
        return grad_input, grad_min, grad_max


class FakeQuantizerBase(Module):
    def __init__(self, int_quant: bool = True, bit:int=4) -> None:
        super().__init__()
        self.lower_bound = Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        self.upper_bound = Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        self.n_bit = Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        # self.n_bit = bit
        self.set_n_bit_manually(bit)
        
        self.bit2bound = {}
        self.use_bit2bound = False
        self.size_of_input = None
        
        self.int_quant = int_quant

        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply
        
        self.calibrated = False
        self.one_direction_search = False
        
        global total_num 
        total_num += 1
        

    def set_int_quant(self, enable: bool):
        self.int_quant = enable

    def set_require_grad(self, enable_lb: bool, enable_up: bool, enable_nbit: bool):
        self.lower_bound.requires_grad = enable_lb
        self.upper_bound.requires_grad = enable_up
        # self.n_bit.requires_grad = enable_nbit

    def set_params_manually(self, lb: Tensor, ub: Tensor, n_bit: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)
        # self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)
        
    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
    
    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.device
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)

    def set_n_bit_manually(self, n_bit):
        device = self.n_bit.device
        self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)


class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self,bit=4) -> None:
        super(FakeQuantizerWeight, self).__init__(bit=bit)

    def forward(self, x:torch.Tensor):
        if not self.calibrated:
            lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)
            self.calibrated = True
            return x
        if self.size_of_input is None:
            self.size_of_input = x.numel()
        
        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound
    
        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        # # r = self.clip(self.round((c - self.lower_bound) / s - 0.5, 0, torch.pow(2, n_bits)-1))
        # r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self,bit=4) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super(FakeQuantizerAct, self).__init__(bit=bit)

        self.running_stat = False # initial boundary
        self.first_iter = False # initial boundary
        self.dynamic = False
        self.beta = 0.995
        self.identity = False

    def forward(self, x):
        if not self.calibrated:
            lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)
            self.calibrated = True
            return x

        if self.size_of_input is None:
            self.size_of_input = x.numel()  

        if self.identity:
            return x
        
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / (torch.pow(2, n_bits) - 1)

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')
            
        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound
        
        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        #     r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound
    
##########################################################
##########################################################


class FakeQuantizerWeightParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / self.n_bit

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class FakeQuantizerActParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super().__init__()

        self.running_stat = True # initial boundary
        self.first_iter = True # initial boundary
        self.dynamic = False
        self.beta = 0.995

    def forward(self, x):
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
            n_bits = self.n_bit
            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / n_bits

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / n_bits

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class QuantBase(Module):
    def __init__(self,config):
        super().__init__()
        self.quant = True
        self.bit = config['bit']
        self.weight_quantizer = FakeQuantizerWeight(self.bit)
        self.act_quantizer = FakeQuantizerAct(self.bit)
        # if config['param'] == 1:
        #     self.weight_quantizer = FakeQuantizerWeight()
        #     self.act_quantizer = FakeQuantizerAct()
        # elif config['param'] == 2:
        #     self.weight_quantizer = FakeQuantizerWeightParam2()
        #     self.act_quantizer = FakeQuantizerActParam2()

    def get_weight_quantizer(self):
        return self.weight_quantizer

    def get_act_quantizer(self):
        return self.act_quantizer

    def set_quant_flag(self, enable: bool):
        self.quant = enable

    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.weight_quantizer.set_require_grad(enable,enable, enable)
        self.act_quantizer.set_require_grad(enable,enable, enable)

    def set_weight_bias_grad(self, enable: bool):
        self.weight.requires_grad = enable
        if self.bias:
            self.bias.requires_grad = enable

    def get_quant_weight_bias(self):
        quant_weight = self.weight_quantizer(self.weight)

        return (quant_weight, self.bias)


class QuantLinear(QuantBase):
    def __init__(self,config):
        super().__init__(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.act_quantizer.set_params_lb_manually(min_value)
        self.act_quantizer.set_params_ub_manually(max_value)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.in_features
        self.out_feature = linear.out_features

        self.weight = Parameter(linear.weight.data.clone())

        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data.clone())
        else:
            self.bias = linear.bias

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)

class QuantLinearQKV(Module):
    def __init__(self,config):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)
        
    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)
        
    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)


    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)
    

    def get_quant_weight_bias(self):
        w_q,b_q = self.q.get_quant_weight_bias()
        w_k,b_k = self.k.get_quant_weight_bias()
        w_v,b_v = self.v.get_quant_weight_bias()
        
        quant_weight = torch.cat([w_q, w_k, w_v],dim=0)
        if b_q is not None:
            bias = torch.cat([b_q,b_k,b_v])

        return (quant_weight, bias)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.in_features 
        self.out_feature = linear.out_features // 3
        
        linear_q = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_k = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_v = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        
        
        linear_q.weight.data,linear_k.weight.data,linear_v.weight.data = linear.weight.data.clone().reshape(3,self.in_feature,self.out_feature)
        
        
        if linear.bias is not None:
            linear_q.bias.data, linear_k.bias.data, linear_v.bias.data = linear.bias.data.clone().reshape(3,self.out_feature)
    
        self.q.set_param(linear_q)
        self.k.set_param(linear_k)
        self.v.set_param(linear_v)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # print('qsize',q.size())
        # print('ksize',k.size())
        # print('vsize',v.size())
        return torch.cat([q,k,v], dim=-1)
    
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)


class QuantConv2d(QuantBase):
    def __init__(self,config):
        super().__init__(config)

    def set_param(self, conv: Conv2d):
        """
        must be called before forward.
        """

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.conv_kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        }
        self.weight = Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data.clone())
        else:
            self.bias = conv.bias

    def forward(self, x):
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.conv_kwargs)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias doesn't need to be quant

        return F.conv2d(quant_act, quant_weight, self.bias, **self.conv_kwargs)


if __name__ == "__main__":

    def differentiable_test():
        x = Tensor((1.0, 2.0, 3.0, 4.0, 5.0))
        lb = Tensor((2.0,))
        ub = Tensor((4.0,))

        n = Tensor((2.0,))

        x.requires_grad = True
        lb.requires_grad = True
        ub.requires_grad = True
        n.requires_grad = True

        s = (ub - lb) / (torch.pow(2, n) - 1)
        print(f"s:{s}")
        round = Differentiable_Round.apply
        clip = Differentiable_Clip.apply

        clipped = clip(x, lb, ub)
        subbed = clipped - lb
        normd = subbed / s
        roundded = round(normd)
        unnormed = s * roundded + lb

        x_hat_sum = unnormed.sum()
        x_hat_sum.backward()

        print("clipped", clipped)
        print("subbed", subbed)
        print(subbed[2] / s)
        print(type(s))
        print("normd", normd, float(normd[2]))
        print("roundded", roundded)
        print("unnormed", unnormed)
        print("=" * 10)
        print(x_hat_sum)
        print(x.grad)
        print(lb.grad)
        print(ub.grad)
        print(n.grad)

    def linear_test():
        l = Linear(10, 10, True)
        l_nb = Linear(10, 10, False)

        print(l.weight)
        print(l.bias)

        print(l_nb.weight)
        print(l_nb.bias)
        if l_nb.bias:
            print(1)
        else:
            print(2)

    # differentiable_test()
    # linear_test()

    def conv_test():
        c = Conv2d(2, 6, (3, 4), 2, 1, bias=True)
        c_nb = Conv2d(3, 6, 3, 2, 1, bias=False)

        print(c.weight.shape)
        # n_kernal(out_channels),in_channels, kerner_shape_H, kernel_shape_W
        print(c.bias.shape)  # n_kernal(out_channels)

        print(c_nb.weight.shape)
        print(c_nb.bias)

        # print(c.)

    def quant_sim_test():
        torch.manual_seed(1234)
        l = Linear(10, 10, True)

        l_q = QuantLinear()
        l_q.set_param(l)
        l_q.set_quant_flag(False)

        input = torch.randn((16, 3, 10))
        out1 = l(input)
        out2 = l_q(input)

        # No quant: tensor(0., grad_fn=<SumBackward0>)
        print("No quant:", (out1 - out2).sum())

        l_q.set_quant_flag(True)
        act_min_val, act_max_val = input.min(), input.max()
        weight_min_val, weight_max_val = l_q.weight.min(), l_q.weight.max()

        l_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 4
        )
        l_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 4)

        out1 = l(input)
        out2 = l_q(input)

        # 4bit quant: tensor(0.7627, grad_fn=<SumBackward0>)
        print("4bit quant:", (out1 - out2).sum())

        l_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 8
        )
        l_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 8)

        out1 = l(input)
        out2 = l_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("8bit quant:", (out1 - out2).sum())

        ######################################
        ######################################
        c = Conv2d(3, 8, 3)

        c_q = QuantConv2d()
        c_q.set_param(c)
        c_q.set_quant_flag(False)

        input = torch.randn((16, 3, 15, 15))
        c_q.set_quant_flag(False)
        out1 = c(input)
        out2 = c_q(input)

        # No quant: tensor(0., grad_fn=<SumBackward0>)
        print("No quant:", (out1 - out2).sum())

        c_q.set_quant_flag(True)
        act_min_val, act_max_val = input.min(), input.max()
        weight_min_val, weight_max_val = c_q.weight.min(), c_q.weight.max()

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 4
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 4)

        out1 = c(input)
        out2 = c_q(input)

        # 4bit quant: tensor(0.7627, grad_fn=<SumBackward0>)
        print("4bit quant:", (out1 - out2).sum())

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 8
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 8)

        out1 = c(input)
        out2 = c_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("8bit quant:", (out1 - out2).sum())

        # import matplotlib.pyplot as plt

        # plt.subplot(1, 2, 1)
        # plt.hist(c.weight.reshape(-1).detach().numpy(), bins=40)
        # plt.subplot(1, 2, 2)
        # plt.hist(c_q.get_quant_weight_bias()[0].reshape(-1).detach().numpy(), bins=40)
        # plt.show()

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 16
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 16)

        out1 = c(input)
        out2 = c_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("16bit quant:", (out1 - out2).sum())

    # quant_sim_test()

    def replace_test():

        class test_module(Module):
            def __init__(self):
                super().__init__()

                self.c1 = Conv2d(3, 16, 3)
                self.act = ReLU()
                self.l = Linear(2704, 15)

            def forward(self, x):
                x = self.c1(x)
                x = self.act(x).view(x.size(0), -1)
                print(x.shape)
                x = self.l(x)

                return x

        input = torch.randn((16, 3, 15, 15))

        m = test_module()
        out1 = m(input)

        for name, module in m.named_modules():
            if isinstance(module, Linear):
                new_module = QuantLinear()
                new_module.set_param(module)
                module = new_module
                module.set_quant_flag(True)
            elif isinstance(module, Conv2d):
                new_module = QuantConv2d()
                new_module.set_param(module)
                module = new_module
                module.set_quant_flag(True)

        # for name, module in m.named_modules():
        #     if isinstance(module, Linear):
        #         print(1)
        #     elif isinstance(module, Conv2d):
        #         print(2)
        print(m)

        out2 = m(input)

        print("8bit quant:", (out1 - out2).sum())
    # replace_test()
    def linear_shape_test():
        l = Linear(2, 6, True)
        print(l.weight)
        print(l.weight.size())
        print(l.bias.size())
        print(l.weight.reshape(3,2,2))
        a,b,c = l.weight.reshape(3,2,2)
        print(a,b,c)
    # linear_shape_test()
    
    def qkv_equal_test():
        torch.random.manual_seed(3407)
        import torch.nn.functional as F
        config = {'param':1}
        in_feature = 30
        out_feature = 90
        l = Linear(3, 9, False)
        l_q = QuantLinearQKV(config)
        l_qq = QuantLinear(config)
        print(l.weight.size())
        
        l_qq.set_param(l)
        l_q.set_param(l)
        
        l_qq.set_quant_flag(False)
        l_q.q.set_quant_flag(False)
        l_q.k.set_quant_flag(False)
        l_q.v.set_quant_flag(False)
        
        input = torch.randn(5, 3)
        
        out1 = l(input)
        out2, q, k ,v = l_q(input)
        out3 = l_qq(input)
        
        q_1, k_1, v_1 = out1.reshape(5,3,3).permute(1,0,2)
        
        print(out1.size())
        print(out2.size())
        print(out3.size())
        print(F.l1_loss(out1, out2))
        print(F.l1_loss(out1, out3))
        print(F.l1_loss(q, q_1))
        print(F.l1_loss(q, k_1))
        print(F.l1_loss(q, v_1))
        print(F.l1_loss(k, k_1))
        print(F.l1_loss(v, v_1))
        
        weight_raw = l.weight.data
        weight_q = l_q.q.weight
        weight_k = l_q.k.weight
        weight_v = l_q.v.weight
        print('weight loss')
        print(F.l1_loss(weight_q, weight_raw[:3,:]))
        print(F.l1_loss(weight_k, weight_raw[3:6,:]))
        print(F.l1_loss(weight_v, weight_raw[6:,:]))
        
        out_ma_1 = input @ weight_q.transpose(0,1)
        print('mannul')
        print(F.l1_loss(out_ma_1, q))
        print(F.l1_loss(out_ma_1, q_1))
        print(F.l1_loss(out_ma_1, out1[:,:3]))
        print(F.l1_loss(out_ma_1, out1[:,3:6]))
        print(F.l1_loss(out_ma_1, out1[:,6:]))
        
        print('linear self')
        out_ma = input @ weight_raw.transpose(0,1) # 这就是linear的计算方式
        out_q_real = out_ma[:,:3] # linear得到的q
        out_q = input @ weight_raw[:3,:].transpose(0,1)
        out_q1 = input @ weight_raw.transpose(0,1)[:,:3]
        out_q2 = input @ weight_raw.reshape(3, 3,3 ).permute(1, 0, 2)[0].transpose(0,1)
        print(F.l1_loss(out_ma, out1))
        print(F.l1_loss(out_q_real, out_q))
        print(F.l1_loss(out_q_real, out_q1))
        print(F.l1_loss(out_q_real, out_q2))
        
        print('detail')
        
        qkv = l(input).reshape(5, 3, 3).permute(1, 0, 2)
        
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        print(F.l1_loss(q, out_q_real))
        print('weight show')
        print(weight_raw.transpose(0,1))
        print(weight_raw[:3,:].transpose(0,1))
        
        print(input@weight_raw.transpose(0,1))
        print((input@weight_raw.transpose(0,1))[:,:3])
        print(input@weight_raw[:3,:].transpose(0,1))
        
        w1 = weight_raw.transpose(0,1)[:,:3]
        w2 = weight_raw[:3,:].transpose(0,1)
        
        print(F.l1_loss(w1, w2))
        print(F.l1_loss(input@w1, input@w2))
        print(F.l1_loss(input@w1, q))
        print(F.l1_loss(input@w2, q))
        
        print(F.l1_loss((input@(weight_raw.transpose(0,1))[:,:3]), input@(weight_raw[:3,:].transpose(0,1))))
        
    # qkv_equal_test()
    
    def qkv_equal_gpt():
        import torch
        import torch.nn as nn

        # 定义一个函数来拆分线性层成三个线性层
        def split_linear_layer(linear_layer, split_size):
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            q_linear = nn.Linear(in_features, split_size)
            k_linear = nn.Linear(in_features, split_size)
            v_linear = nn.Linear(in_features, split_size)
            q_linear.weight.data = linear_layer.weight.data[:split_size, :]
            k_linear.weight.data = linear_layer.weight.data[split_size:2*split_size, :]
            v_linear.weight.data = linear_layer.weight.data[2*split_size:, :]
            q_linear.bias.data = linear_layer.bias.data[:split_size]
            k_linear.bias.data = linear_layer.bias.data[split_size:2*split_size]
            v_linear.bias.data = linear_layer.bias.data[2*split_size:]
            return q_linear, k_linear, v_linear

        # 创建一个示例线性层
        linear_layer = nn.Linear(100, 300)

        # 将线性层拆分成三个线性层
        q_linear, k_linear, v_linear = split_linear_layer(linear_layer, 100)

        # 创建一些输入数据
        batch_size = 16
        input_data = torch.randn(batch_size, 100, dtype=torch.float64)  # 使用 torch.float64

        # 使用拆分前的线性层进行前向传播
        output_before_split = linear_layer(input_data)

        # 使用拆分后的线性层进行前向传播
        q_output, k_output, v_output = q_linear(input_data), k_linear(input_data), v_linear(input_data)
        output_after_split = torch.cat([q_output, k_output, v_output], dim=1)

        # 计算 L1 损失来验证结果一致性
        l1_loss = nn.L1Loss()
        loss = l1_loss(output_before_split, output_after_split)

        print("L1 损失:", loss.item())

    # qkv_equal_gpt()
    
    def param_test():
        torch.random.manual_seed(3407)
        import torch.nn.functional as F
        
        config = {'param':1}
        l_raw = Linear(10,20)
        l_q = QuantLinear(config)
        l_q.set_param(l_raw)
        
        input = torch.randn(5, 10)
        
        l_q.set_quant_flag(False)
        
        out1 = l_raw(input)
        out2 = l_q(input)
        
        print(F.l1_loss(out1, out2))
        
        # print(l_q.named_parameters())
        for name, param in l_q.named_parameters():
            print(name)
        
        
    param_test()