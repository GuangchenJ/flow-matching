import torch
import torch.nn as nn
import lightning


class OTFlowMatching:

    def __init__(self, sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
        """ Conditional Flow
        """
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) +
             torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 -
                                                                      self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x_0), 一开始是服从均值为 0、方差为 1 的正态分布的随机数
        x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
        # psi_t 的微分就是 v
        d_psi = x_1 - (1 - self.sig_min) * x_0
        # 利用 均方误差 MSE 来构建最终的回归形式
        return torch.mean((v_psi - d_psi)**2)
