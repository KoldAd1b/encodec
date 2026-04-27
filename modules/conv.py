import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

CONV_NORMALIZATIONS = ['none',
                       'weight_norm', 'spectral_norm',
                       'time_layer_norm', 'layer_norm', 'time_group_norm'
                       ]

class ConvLayerNorm(nn.LayerNorm):

    """
    LayerNorm expects the normalized dimension last.
    For convolution tensors shaped (B, C, T) or (B, C, ..., T), move time
    before the channel/spatial dimensions, normalize, then restore the layout.
    """

    def __init__(self, 
                 normalized_shape, 
                 **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x

def apply_parameterization_norm(module, norm):

    """
    Apply weight-based normalization to a convolution module when requested.
    """
    
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # Activation/output normalizations are applied separately after the conv.
        return module
    
def get_norm_module(module, norm, **norm_kwargs):
    """
    Return the output normalization paired with a convolution module.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":

        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

class NormConv1d(nn.Module):
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.conv = apply_parameterization_norm(
            nn.Conv1d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.conv, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))
    
class NormConv2d(nn.Module):
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.conv = apply_parameterization_norm(
            nn.Conv2d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.conv, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))
    
class NormTransposeConv1d(nn.Module):
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.convtr = apply_parameterization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.convtr, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))

class NormTransposeConv2d(nn.Module):
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.convtr = apply_parameterization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.convtr, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))

def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):

    """
    Return the extra right padding needed so every strided convolution window
    is complete. This makes the output length ceil(input_length / stride)
    instead of dropping a partial final window.
    """
    
    length = x.shape[-1]

    # Fractional frame counts indicate a final incomplete convolution window.
    n_frames = (length - kernel_size + padding_total) / stride + 1

    # Compute the input length required to realize ceil(n_frames) full windows.
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)

    return ideal_length - length

def pad1d(x, paddings, mode="zero", value=0):
    """
    Pad a 1D signal, with a fallback for reflect padding on very short inputs.
    Reflection requires the input to be longer than the padding size, so short
    inputs are temporarily zero-padded on the right before reflection.
    """
    length = x.shape[1]
    padding_left, padding_right = paddings

    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x,(0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

def unpad1d(x, paddings):
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]

class SConv1d(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 norm="none",
                 norm_kwargs={}, 
                 pad_mode="reflect"):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm = norm 
        self.pad_mode = pad_mode

        # Padding is applied manually in forward().
        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride, 
            dilation=dilation, groups=groups, bias=bias, 
            norm=norm, norm_kwargs=norm_kwargs
        )

    def forward(self, x):

        batch, channels, seq_len = x.shape

        # Dilation increases the time span covered by a kernel.
        # Example: kernel_size=3, dilation=2 spans 5 samples.
        kernel_size = (self.kernel_size - 1) * self.dilation + 1

        # Choose padding so the strided conv produces roughly T / stride frames:
        # output = floor((T + padding_total - kernel_size) / stride) + 1.
        padding_total = kernel_size - self.stride

        # Add right padding when T is not stride-aligned, so the last partial
        # window becomes a full convolution window instead of being dropped.
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, self.stride, padding_total)
 
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        return self.conv(x)

class SConvTranspose1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 norm="none", 
                 norm_kwargs={}):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm

        self.convtr = NormTransposeConv1d(
            in_channels, out_channels, kernel_size, stride,
            norm=norm, norm_kwargs=norm_kwargs
        )

    def forward(self, x):

        padding_total = self.kernel_size - self.stride

        y = self.convtr(x)

        # Remove the deterministic padding paired with SConv1d. The dynamic
        # extra padding depends on the original input length and is handled by
        # final output trimming at the model level.
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        y = unpad1d(y, (padding_left, padding_right))

        return y

if __name__ == "__main__":

    sconv = SConv1d(in_channels=16, 
                    out_channels=32,
                    kernel_size=11,
                    stride=2)
    
    rand = torch.randn(4,16,100)
    
    print(rand.shape)
    conv_out = sconv(rand)
    print(conv_out.shape)

    tsconv = SConvTranspose1d(in_channels=32, 
                              out_channels=16, 
                              kernel_size=11, 
                              stride=2)
    tsconv_out = tsconv(conv_out)
    print(tsconv_out.shape)
