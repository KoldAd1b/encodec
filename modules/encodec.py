import torch
import torch.nn as nn
import random
import accelerate 
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

from .seanet import SEANetEncoder, SEANetDecoder
from .quantizer import ResidualVectorQuantization

@dataclass
class EnCodecConfig:

    channels: int = 1                                                          
    dimension: int = 128
    n_filters: int = 32
    n_residual_layers: int = 1
    ratios: Tuple[int, ...] = (8, 5, 4, 2)
    activation: str = "Snake"
    activation_params: Dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})
    final_activation: str = None
    final_activation_params: Dict[str, Any] = field(default_factory=dict)
    norm: str = "weight_norm"
    norm_params: Dict[str, Any] = field(default_factory=dict)
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    pad_mode: str = "reflect"
    true_skip: bool = False
    compress: int = 2
    lstm: int = 2
    lstm_bidirectional: bool = True

    num_quantizers: int = 8
    codebook_dim: int = None
    codebook_size: int = 1024
    decay: float = 0.99
    kmeans_init: bool = True
    kmeans_iters: int = 50
    threshold_ema_dead_code: int = 2
    commit_weight: float = 1

class EncodecModel(nn.Module):
    def __init__(self, config, accelerator=None):
        
        super().__init__()

        self.config = config
        self.num_quantizers = config.num_quantizers
        self.accelerator = accelerator

        self.encoder = SEANetEncoder(
            channels=config.channels, dimension=config.dimension, n_filters=config.n_filters, 
            n_residual_layers=config.n_residual_layers, ratios=config.ratios,
            activation=config.activation, activation_params=config.activation_params,
            norm=config.norm, norm_params=config.norm_params,
            kernel_size=config.kernel_size, last_kernel_size=config.last_kernel_size, 
            residual_kernel_size=config.residual_kernel_size, dilation_base=config.dilation_base, 
            pad_mode=config.pad_mode, true_skip=config.true_skip, compress=config.compress, lstm=config.lstm, 
            lstm_bidirectional=config.lstm_bidirectional
        )

        self.quantizer = ResidualVectorQuantization(
            num_quantizers=config.num_quantizers, dim=self.encoder.dimension, 
            codebook_size=config.codebook_size, codebook_dim=config.codebook_dim, 
            decay=config.decay, kmeans_init=config.kmeans_init, 
            kmeans_iters=config.kmeans_iters, threshold_ema_dead_code=config.threshold_ema_dead_code, 
            commitment_weight=config.commit_weight, accelerator=accelerator
        )

        self.decoder = SEANetDecoder(
            channels=config.channels, dimension=config.dimension, n_filters=config.n_filters, 
            n_residual_layers=config.n_residual_layers, ratios=config.ratios,
            activation=config.activation, activation_params=config.activation_params,
            final_activation=config.final_activation, final_activation_params=config.final_activation_params,
            norm=config.norm, norm_params=config.norm_params, 
            kernel_size=config.kernel_size, last_kernel_size=config.last_kernel_size,
            residual_kernel_size=config.residual_kernel_size, dilation_base=config.dilation_base,
            pad_mode=config.pad_mode, true_skip=config.true_skip, compress=config.compress, lstm=config.lstm, 
            lstm_bidirectional=config.lstm_bidirectional
        )

    def _is_distributed(self):
        return self.accelerator is not None and self.accelerator.num_processes > 1

    def normalize(self, x):

        """
        Per sample loudness normalization, we rescale the signal so each sample 
        has unit RMS (root mean squared)
        """
        
        batch_size, channels, seq_len = x.shape

        mono = x
        if channels > 1:
            mono = x.mean(dim=1, keepdim=True)

        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
        scale = 1e-8 + volume

        x = x / scale

        return x, scale

    def denormalize(self, x, scale):
        return x * scale

    def forward(self, x):

        batch_size, channels, seq_len = x.shape

        x, scale = self.normalize(x)
        encoded = self.encoder(x)

        num_books_to_use = torch.tensor(random.randint(0, self.num_quantizers), device=x.device)

        if self._is_distributed():
            num_books_to_use = accelerate.utils.broadcast(num_books_to_use, from_process=0)
        
        quantized_out, codebook_indices, out_losses = self.quantizer(encoded, num_books_to_use)
        decoded = self.decoder(quantized_out)

        decoded = self.denormalize(decoded, scale)
        decoded = decoded[:, :, :seq_len]

        return {"encoder_out": encoded, 
                "quantized": quantized_out, 
                "quantizer_loss": torch.mean(out_losses),
                "quantizer_losses": out_losses,
                "codebook_indices": codebook_indices,
                "num_books_to_use": num_books_to_use,
                "decoded": decoded}

    @torch.no_grad()
    def tokenize(self, x):  

        x, scale = self.normalize(x)
        encoded = self.encoder(x)
        tokens = self.quantizer.encode(encoded)

        return tokens, scale
        

    @torch.no_grad()
    def decode(self, indices, scale, max_len=None):
        quantized = self.quantizer.decode(indices)
        decoded = self.decoder(quantized)
        decoded = self.denormalize(decoded, scale)

        if max_len is not None:
            decoded = decoded[:, :, :max_len]
        
        decoded = torch.clamp(decoded, -1.0, 1.0)

        return decoded
    
    @torch.no_grad()
    def passthrough(self, x):
        tokens, scale = self.tokenize(x)

        reconstruction = self.decode(tokens, scale, max_len=x.shape[-1])

        return reconstruction
        

if __name__ == "__main__":
    rand = torch.randn(2,1,5)
    model = EncodecModel()
    out, scale = model.normalize(rand)
    denormed = model.denormalize(out, scale)

    print(rand.mean(), rand.std())
    print(out.mean(), out.std())
    print(denormed.mean(), denormed.std())
