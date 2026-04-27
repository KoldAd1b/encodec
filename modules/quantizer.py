import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator

def ema_inplace(moving_average, new, decay):
    moving_average.data.mul_(decay).add_(new, alpha=(1-decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def sample_vectors(samples, num):
    num_samples = samples.shape[0]
    device = samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)
    
    return samples[indices]

def kmeans(samples, num_clusters, num_iters):
    """Initialize codebook embeddings with k-means."""

    dim = samples.shape[-1]
    dtype = samples.dtype
    
    means = sample_vectors(samples, num_clusters)   

    for _ in range(num_iters):

        samples_, means_ = samples.unsqueeze(1), means.unsqueeze(0)
        diffs = samples_ - means_
        dists = -1 * (diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)

        zero_mask = (bins == 0)
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)

        repeated_buckets = buckets.unsqueeze(-1).repeat(1, dim)
        new_means.scatter_add_(dim=0, index=repeated_buckets, src=samples)
        new_means = new_means / bins_min_clamped.unsqueeze(-1)

        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    def __init__(self, 
                 dim, 
                 codebook_size, 
                 kmeans_init=False,
                 kmeans_iter=50, 
                 decay=0.99, 
                 epsilon=1e-5, 
                 threashold_ema_dead_code=2,
                 accelerator=None):
        
        super().__init__()

        self.decay = decay
        self.codebooks_size = codebook_size
        self.kmeans_iters = kmeans_iter
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threashold_ema_dead_code
        self.accelerator = accelerator
        
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def _is_distributed(self):
        return self.accelerator is not None and self.accelerator.num_processes > 1

    def _broadcast_buffers(self):
        if not self._is_distributed():
            return
            
        for buffer in self.buffers():
            buffer.data = accelerate.utils.broadcast(buffer.data, from_process=0)

    @torch.jit.ignore
    def init_embed_(self, data):

        if self.inited:
            return 
        
        if self._is_distributed():

            all_data = self.accelerator.gather(data)
            
            if self.accelerator.is_main_process:
                embed, cluster_size = kmeans(all_data, self.codebooks_size, self.kmeans_iters)
                self.embed.data.copy_(embed)
                self.embed_avg.data.copy_(embed.clone())
                self.cluster_size.data.copy_(cluster_size)
            
            self.accelerator.wait_for_everyone()
            self._broadcast_buffers()
        
        else:

            embed, cluster_size = kmeans(data, self.codebooks_size, self.kmeans_iters)
            self.embed.data.copy_(embed)
            self.embed_avg.data.copy_(embed.clone())
            self.cluster_size.data.copy_(cluster_size)
        
        self.inited.data.copy_(torch.Tensor([True]))

        if self._is_distributed():
            self.inited.data = accelerate.utils.broadcast(self.inited.data, from_process=0)

    def replace_(self, samples, mask):
        
        modified_codebook = torch.where(
            mask.unsqueeze(-1), sample_vectors(samples, self.codebooks_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)
    
    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        
        expired_codes = (self.cluster_size < self.threshold_ema_dead_code)

        if not torch.any(expired_codes):
            return
        
        batch_samples = einops.rearrange(batch_samples, "... d -> (...) d")

        if not self._is_distributed() or self.accelerator.is_main_process:
            self.replace_(batch_samples, mask=expired_codes)

        if self._is_distributed():
            self.accelerator.wait_for_everyone()
            self.embed.data = accelerate.utils.broadcast(self.embed.data, from_process=0)

    def preprocess(self, x):
        x = einops.rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):

        embed = self.embed.t()

        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices

        return embed_ind
    
    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.reshape(*shape[:-1])
     
    def dequantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed)

    @torch.no_grad()
    def encode(self, x):
        
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)

        return embed_ind
    
    @torch.no_grad()
    def decode(self, embed_ind):
        return self.dequantize(embed_ind)
    
    def forward(self, x):
        shape, dtype = x.shape, x.dtype

        x = self.preprocess(x)
        self.init_embed_(x)

        embed_ind = self.quantize(x) 
        embed_onehot = F.one_hot(embed_ind, self.codebooks_size).type(dtype)

        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantized = self.dequantize(embed_ind)

        if self.training:

            if self._is_distributed():
                batch_samples = self.accelerator.gather(x)
            else:
                batch_samples = x
            
            self.expire_codes_(batch_samples)

            local_cluster_count = embed_onehot.sum(0)
            local_embed_sum = x.t() @ embed_onehot

            if self._is_distributed():
                global_cluster_count = self.accelerator.reduce(local_cluster_count, reduction="sum")
                global_embed_sum = self.accelerator.reduce(local_embed_sum, reduction="sum")
            else:
                global_cluster_count = local_cluster_count
                global_embed_sum = local_embed_sum
            
            ema_inplace(self.cluster_size, global_cluster_count, self.decay)
            ema_inplace(self.embed_avg, global_embed_sum.t(), self.decay)

            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebooks_size, self.epsilon)
                * self.cluster_size.sum()
            )

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            if self._is_distributed():
                self._broadcast_buffers()
            
            if self._is_distributed():
                self.accelerator.wait_for_everyone()
        
        return quantized, embed_ind

class VectorQuantization(nn.Module):
    def __init__(self, 
                 dim, 
                 codebook_size, 
                 codebook_dim=None, 
                 decay=0.99, 
                 epsilon=1e-5,
                 kmeans_init=True, 
                 kmeans_iters=50, 
                 threshold_ema_dead_code=2, 
                 commitment_weight=1,
                 accelerator=None):
        
        super().__init__()

        self.decay = decay
        self.codebooks_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.accelerator = accelerator
        self.commitment_weight = commitment_weight

        codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = codebook_dim != dim
        self.proj_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.proj_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()

        self._codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size, 
                                           kmeans_init=kmeans_init, kmeans_iter=kmeans_iters,
                                           decay=decay, epsilon=epsilon, 
                                           threashold_ema_dead_code=threshold_ema_dead_code, 
                                           accelerator=accelerator)
    
    @torch.no_grad()
    def encode(self, x):
        x = einops.rearrange(x, "b d n -> b n d")
        x = self.proj_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in
    
    @torch.no_grad()
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.proj_out(quantize)
        quantize = einops.rearrange(quantize, "b n d -> b d n")
        return quantize
    
    def forward(self, x):
        device = x.device

        x = einops.rearrange(x, "b d n -> b n d")
        x = self.proj_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()
        
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight
        
        quantize = self.proj_out(quantize)
        quantize = einops.rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss

class ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers, **kwargs):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    @torch.no_grad()
    def encode(self, x, n_q=None):
        
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)

        out_indices = torch.stack(all_indices)
        return out_indices

    @torch.no_grad()
    def decode(self, q_indices):

        quantized_out = torch.tensor(0.0, device=q_indices.device)

        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
            
        return quantized_out
    
    def forward(self, x, n_q=None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:

            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices)
        
        return quantized_out, out_indices, out_losses

def test_kmeans():
    import matplotlib.pyplot as plt

    true_centers = torch.tensor([
        [-5.0, -5.0],
        [0.0, 5.0],
        [5.0, -2.0]
    ])

    num_points_per_cluster = 300
    dim = 2
    samples_list = []

    for center in true_centers:
        points = center + torch.randn(num_points_per_cluster, dim)
        samples_list.append(points)

    samples = torch.cat(samples_list, dim=0)

    means, bins = kmeans(samples, 3, 20)

    plt.figure(figsize=(6,6))

    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5)

    plt.scatter(means[:, 0], means[:, 1],
                color="red",
                s=200,
                marker="x",
                linewidths=3)

    plt.title("K-means clustering result")
    plt.show()


def test_codebook():
    accelerator = Accelerator()
    
    dim = 128
    codebook_size = 16
    seq_len = 100
    batch_size = 8
    
    codebook = EuclideanCodebook(
        dim=dim,
        codebook_size=codebook_size,
        kmeans_init=True,
        accelerator=accelerator,
        threashold_ema_dead_code=10
    )
    
    codebook = codebook.to(accelerator.device)
    codebook.train()
    
    accelerator.print(f"\n{'='*50}")
    accelerator.print(f"Testing EuclideanCodebook with DDP")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Current process: {accelerator.process_index}")
    accelerator.print(f"{'='*50}\n")
    
    accelerator.print("Step 1: Testing k-means initialization...")
    x = torch.randn(batch_size, seq_len, dim).to(accelerator.device)
    
    quantized, indices = codebook(x)
    
    accelerator.wait_for_everyone()
    

    if accelerator.is_main_process:
        embed_sum = codebook.embed.sum().item()
        cluster_size_sum = codebook.cluster_size.sum().item()
        accelerator.print(f"✓ Initialization complete")
        accelerator.print(f"  Embed sum: {embed_sum:.4f}")
        accelerator.print(f"  Cluster size sum: {cluster_size_sum:.4f}")
    
    accelerator.wait_for_everyone()
    
    accelerator.print("\nStep 2: Testing training updates...")
    for step in range(3):
        x = torch.randn(batch_size, dim).to(accelerator.device)
        quantized, indices = codebook(x)
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            embed_sum = codebook.embed.sum().item()
            cluster_size_sum = codebook.cluster_size.sum().item()
            accelerator.print(f"  Step {step+1}: embed_sum={embed_sum:.4f}, cluster_size_sum={cluster_size_sum:.4f}")
    
    accelerator.print("\nStep 3: Verifying buffer synchronization...")
    
    if accelerator.num_processes > 1:
        embed_gathered = accelerator.gather(codebook.embed)
        cluster_size_gathered = accelerator.gather(codebook.cluster_size)
        
        if accelerator.is_main_process:

            embed_split = embed_gathered.view(accelerator.num_processes, codebook_size, dim)
            cluster_size_split = cluster_size_gathered.view(accelerator.num_processes, codebook_size)
            
            embed_diff = (embed_split[0].unsqueeze(0) - embed_split[1:]).abs().max().item()
            cluster_diff = (cluster_size_split[0].unsqueeze(0) - cluster_size_split[1:]).abs().max().item()
            
            accelerator.print(f"Max embedding difference across GPUs: {embed_diff:.2e}")
            accelerator.print(f"Max cluster_size difference across GPUs: {cluster_diff:.2e}")
            
            if embed_diff < 1e-5 and cluster_diff < 1e-5:
                accelerator.print("SUCCESS: All buffers are properly synchronized!")
            else:
                accelerator.print("FAILED: Buffers are not synchronized!")
    else:
        accelerator.print("Single process mode - no synchronization needed")
    
    accelerator.print(f"\n{'='*50}\n")
    
    accelerator.end_training()

if __name__ == "__main__":
    test_codebook()
