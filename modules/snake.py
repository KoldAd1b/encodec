import torch
import triton
import triton.language as tl
import torch.nn as nn

class SlowSnake(nn.Module):
    """Acitvation function proposed in https://arxiv.org/abs/2006.08195
    and used in SpeechTokenizer, BigvGAN and others architectures 
    to inject an inductive bias of periodicity into the architecture"""
    
    def __init__(self, 
                 in_features, 
                 alpha=1.0, 
                 params_trainable=True,
                 eps=1e-6):
        super().__init__()

        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha, requires_grad=params_trainable)
       
    def forward(self, x):

        ### alpha is size (in_features, ), our data is (B x in_features x T), so lets add in dimensions ###
        ### so we can broadcast over the B and T dims ###
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)

        ### forward pass from paper: x + (1 / b) * sin^2(x*a)
        x = x + (1.0 / (alpha + self.eps)) * torch.sin(x * alpha).pow(2)

        return x

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64}, num_warps=2),
        triton.Config({'BLOCK_T': 128}, num_warps=4),
        triton.Config({'BLOCK_T': 256}, num_warps=4),
        triton.Config({'BLOCK_T': 512}, num_warps=8),
        triton.Config({'BLOCK_T': 1024}, num_warps=8),
    ],
    key=['T'],
)
@triton.jit
def snake1d_forward_kernel(
    x_ptr, 
    alpha_ptr, 
    out_ptr,
    B, 
    C, 
    T,
    stride_b, 
    stride_c, 
    stride_t,
    eps,
    BLOCK_T: tl.constexpr,
):
    """
    Grid: (B, C)  — one program per (batch, channel) pair.
    Each program streams over the T dimension in tiles of BLOCK_T.
    """
    b_idx = tl.program_id(0) # which batch are we processing?
    c_idx = tl.program_id(1) # which channel are we processing?

    # Load the scalar alpha for this channel
    a = tl.load(alpha_ptr + c_idx).to(tl.float32)
    safe_a = a + eps
    
    ### Offset to starting timestep ###
    base = b_idx * stride_b + c_idx * stride_c

    ### Loop through timesteps in chunks of BLOCK_T ###
    for t_start in range(0, T, BLOCK_T):

        ### get t offsets ###
        t_offs = t_start + tl.arange(0, BLOCK_T)

        ### Dont spill over ###
        mask = t_offs < T

        ### Offset x_ptr to base and then grab block of pointers ###
        ptrs = x_ptr + base + t_offs * stride_t
        
        ### load data ###
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        
        ### Perform Op ###
        s = tl.sin(safe_a * x)
        out = x + s * s / safe_a
        
        ### Store ###
        tl.store(out_ptr + base + t_offs * stride_t, out, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64}, num_warps=2),
        triton.Config({'BLOCK_T': 128}, num_warps=4),
        triton.Config({'BLOCK_T': 256}, num_warps=4),
        triton.Config({'BLOCK_T': 512}, num_warps=8),
        triton.Config({'BLOCK_T': 1024}, num_warps=8),
    ],
    key=['T'],
    reset_to_zero=['grad_alpha_ptr']
)
@triton.jit
def snake1d_backward_kernel(
    x_ptr, 
    alpha_ptr, 
    grad_out_ptr, 
    grad_x_ptr, 
    grad_alpha_ptr, 
    B, C, T, 
    stride_b, 
    stride_c, 
    stride_t, 
    eps, 
    BLOCK_T: tl.constexpr
):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    a = tl.load(alpha_ptr + c_idx).to(tl.float32)
    safe_a = a + eps

    base = b_idx * stride_b + c_idx * stride_c

    ### alpha was broadcasted over all B, T in the forward pass ###
    ### so we will sum together grad contributions from all B, T ###
    ### in the backward pass ! ###
    acc_ga = tl.zeros((1,), dtype=tl.float32)

    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        mask = t_offs < T
        ptrs = x_ptr + base + t_offs * stride_t

        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(grad_out_ptr + base + t_offs * stride_t, mask=mask, other=0.0).to(tl.float32)
        
        s_ax = tl.sin(safe_a * x)
        s_2ax = tl.sin(2 * safe_a * x)

        # gx = 1 + sin(2ax)
        gx = go * (1.0 + s_2ax)
        tl.store(grad_x_ptr + base + t_offs * stride_t, gx, mask=mask)

        # grad_a: sin(2·a·x)·x/a  -  sin^2(a·x)/a^2
        ga = go * (s_2ax * x / safe_a - (s_ax / safe_a)*(s_ax / safe_a))
        
        # accumulate for this block of timesteps
        acc_ga += tl.sum(tl.where(mask, ga, 0.0))

    ### Sum all grad contributions over time dimension ###
    acc_ga = tl.sum(acc_ga, axis=0)
#
    tl.atomic_add(grad_alpha_ptr + c_idx, acc_ga)
