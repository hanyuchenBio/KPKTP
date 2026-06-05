from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

try:
    from .configuration_ntv3_pretrained import Ntv3PreTrainedConfig  # HF config
except ImportError:
    from .configuration_ntv3_pretrained import Ntv3PreTrainedConfig  # local dev

from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


class LayerNormFP32(nn.LayerNorm):
    """
    LayerNorm that always computes in float32 for numerical stability.

    Standard approaches to fp32 LayerNorm (e.g., `ln(x.float()).to(x.dtype)`) cause
    graph breaks with torch.compile because `.float()` and `.to(x.dtype)` involve
    dynamic dtype inspection that triggers `aten._local_scalar_dense` errors in
    the Inductor backend.

    This implementation avoids graph breaks by:
    - Using `x.to(torch.float32)` with a static target dtype
    - Using `.type_as(x)` for casting back (compiler-friendly)
    - Computing mean/var manually instead of delegating to F.layer_norm

    Drop-in replacement for nn.LayerNorm. Adapted from:
    - PyTorch LayerNorm: https://github.com/pytorch/pytorch/blob/d74f9ecce649b52c52bea857c1c7d1e859ff6826/torch/nn/modules/normalization.py#L106 # noqa: E501
    - Gemma2 RMSNorm (fp32 pattern): https://github.com/huggingface/transformers/blob/a8a22624f5598167eb82a3e0bc4228021e5440f6/src/transformers/models/gemma2/modeling_gemma2.py#L49 # noqa: E501
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        # Compute normalization in fp32
        x_fp32 = input.to(torch.float32)
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
        x_normed = (x_fp32 - mean) * torch.rsqrt(var + self.eps)

        # Apply affine transform in fp32, then cast back
        if self.weight is not None:
            x_normed = x_normed * self.weight.to(torch.float32)
        if self.bias is not None:
            x_normed = x_normed + self.bias.to(torch.float32)

        return x_normed.type_as(input)


def _dtype_from_str(s: str | None) -> torch.dtype:
    s = (s or "float32").lower()
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


@contextmanager
def _autocast_to(
    device_type: str, dtype_str: str | None
) -> Generator[None, None, None]:
    """Enable autocast(bf16) for a block if requested, else disable autocast.
    Does not touch parameter dtypes; only compute autocasts.
    """
    if (dtype_str or "").lower() in {"bfloat16", "bf16"}:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield
    else:
        with torch.autocast(device_type=device_type, enabled=False):
            yield


@dataclass(kw_only=True)
class RotaryEmbeddingConfig:
    rescaling_factor: float | None


class JaxConvTranspose1dSame(nn.Module):
    """JAX-compatible ConvTranspose1d.

    Supports stride=2, odd K, SAME padding with phase control.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        bias: bool = True,
        phase: str = "odd",
    ):
        super().__init__()
        assert stride == 2
        assert kernel_size % 2 == 1
        assert phase in ("even", "odd")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.phase = phase
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.permute(1, 0, 2), a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, Cin, L) -> zero-insert (phase) -> conv1d SAME
        B, C, L = x.shape  # noqa: N806
        up = x.new_zeros(B, C, L * 2)
        if self.phase == "even":
            up[:, :, ::2] = x
        else:
            up[:, :, 1::2] = x
        w = self.weight.permute(1, 0, 2).contiguous()  # (Cout, Cin, K)
        pad = self.kernel_size // 2
        return F.conv1d(up, w, bias=self.bias, stride=1, padding=pad)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, rotary_embedding_config: RotaryEmbeddingConfig):
        super().__init__()
        self.rescaling_factor = rotary_embedding_config.rescaling_factor
        self.upper_freq = 10000.0
        self.dim = dim
        self.register_buffer("inv_freq", self._inv_freq())
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = q.shape[1]  # noqa: N806
        self._ensure_cache(L, q.device, q.dtype)
        cos = self.cos_cached[:, :L]
        sin = self.sin_cached[:, :L]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _inv_freq(self) -> torch.Tensor:
        if self.rescaling_factor is None:
            base = self.upper_freq
        else:
            base = self.upper_freq * (
                self.rescaling_factor ** (self.dim / (self.dim - 2))
            )
        return 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def _ensure_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self.cos_cached is not None
            and seq_len <= self.cos_cached.shape[1]
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = torch.cos(freqs).to(dtype)[None, :, None, :]
        sin = torch.sin(freqs).to(dtype)[None, :, None, :]
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)


class LinearProjectionHeInit(nn.Module):
    def __init__(self, num_heads: int, key_size: int):
        super().__init__()
        in_features = num_heads * key_size
        self.num_heads = num_heads
        self.key_size = key_size
        self.linear = nn.Linear(in_features, in_features, bias=True)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y.reshape((*x.shape[:-1], self.num_heads, self.key_size))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        rotary_embedding_config: RotaryEmbeddingConfig | None = None,
        add_bias_kv: bool = False,
        value_size: int | None = None,
        model_size: int | None = None,
    ):
        super().__init__()
        value_size = value_size if value_size is not None else key_size
        model_size = model_size if model_size is not None else key_size * num_heads
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.model_size = model_size
        self.add_bias_kv = add_bias_kv

        self.query_head = LinearProjectionHeInit(num_heads, key_size)
        self.key_head = LinearProjectionHeInit(num_heads, key_size)
        self.value_head = LinearProjectionHeInit(num_heads, value_size)

        self.mha_output = nn.Linear(num_heads * value_size, model_size)
        nn.init.kaiming_uniform_(self.mha_output.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mha_output.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.mha_output.bias, -bound, bound)

        self._bias_k = None
        self._bias_v = None
        if self.add_bias_kv:
            self.register_buffer("_bias_k", torch.zeros(1, 1, num_heads, key_size))
            self.register_buffer("_bias_v", torch.zeros(1, 1, num_heads, value_size))

        self.rotary_embedding = (
            RotaryEmbedding(key_size, rotary_embedding_config)
            if rotary_embedding_config
            else None
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_weight_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # QKV projections (respect autocast dtype outside)
        q = self.query_head(query)
        k = self.key_head(key)
        v = self.value_head(value)

        if self.add_bias_kv:
            B = k.shape[0]  # noqa: N806
            assert self._bias_k is not None
            assert self._bias_v is not None
            bk = self._bias_k.expand(B, -1, -1, -1)
            bv = self._bias_v.expand(B, -1, -1, -1)
            k = torch.cat((k, bk), dim=1)
            v = torch.cat((v, bv), dim=1)
            if attention_mask is not None:
                mask_bias = torch.ones(
                    attention_mask.shape[:-1] + (1,),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((attention_mask, mask_bias), dim=-1)

        if self.rotary_embedding is not None:
            q, k = self.rotary_embedding(q, k)

        # (B, L, H, D) -> (B, H, L, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ---- Mixed-precision stable path: do attention in fp32 ----
        # keep original autocast dtype to restore later
        out_dtype = q.dtype

        qf = q.float()
        kf = k.float()
        vf = v.float()

        # scale in fp32
        logits = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(self.key_size)

        if attention_mask is not None:
            # attention_mask is bool: True=keep, False=mask
            # use a large negative fp32 value
            logits = torch.where(
                attention_mask.unsqueeze(1), logits, torch.full_like(logits, -1e30)
            )

        if attention_weight_bias is not None:
            logits = logits + attention_weight_bias.float()

        weights_fp32 = F.softmax(logits, dim=-1)  # softmax in fp32
        attn_fp32 = torch.matmul(weights_fp32, vf)  # (B, H, L, D)

        # Back to original autocast dtype for projection
        attn = attn_fp32.to(out_dtype).permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)

        # The output projection: ensure input matches the Linear weight dtype on CPU
        proj_in = attn.reshape((*attn.shape[:-2], -1))
        proj_in = proj_in.to(self.mha_output.weight.dtype)

        out = self.mha_output(proj_in)
        return {"embeddings": out, "attention_weights": weights_fp32.to(out_dtype)}


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        key_size: int | None = None,
        add_bias_kv: bool = False,
        add_bias_fnn: bool = True,
        ffn_activation_name: str = "swish",
        use_glu_in_ffn: bool = True,
        layer_norm_eps: float = 1e-5,
        pre_layer_norm: bool = True,
        ln_dtype: torch.dtype = torch.float32,
        rotary_embedding_config: RotaryEmbeddingConfig | None = None,
    ):
        super().__init__()
        if key_size is None:
            if embed_dim % num_heads != 0:
                raise ValueError("embed_dim must be divisible by num_heads")
            key_size = embed_dim // num_heads

        self._pre_layer_norm = pre_layer_norm
        # keep the exact attribute name you already had
        self._use_glu_in_fnn = use_glu_in_ffn

        self.self_attention_layer_norm = LayerNormFP32(embed_dim, eps=layer_norm_eps)
        self.final_layer_norm = LayerNormFP32(embed_dim, eps=layer_norm_eps)

        self.sa_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_size=key_size,
            add_bias_kv=add_bias_kv,
            model_size=embed_dim,
            rotary_embedding_config=rotary_embedding_config,
        )

        self.fc1 = nn.Linear(
            embed_dim,
            2 * ffn_embed_dim if use_glu_in_ffn else ffn_embed_dim,
            bias=add_bias_fnn,
        )
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=add_bias_fnn)

        if ffn_activation_name == "swish":
            self._ffn_activation_fn = nn.SiLU()
        else:
            # keep parity with your original (no accidental behavior change)
            self._ffn_activation_fn = getattr(torch.nn, ffn_activation_name)

    def self_attention(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_weight_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = self.sa_layer(
            x, x, x, attention_mask, attention_weight_bias
        )
        return result

    def mlp(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.final_layer_norm(x_in) if self._pre_layer_norm else x_in
        if self._use_glu_in_fnn:
            x_lin = self.fc1(x)
            x1, x2 = torch.split(x_lin, x_lin.shape[-1] // 2, dim=-1)
            x = self._ffn_activation_fn(x1) * x2
        else:
            x = self._ffn_activation_fn(self.fc1(x))
        x = self.fc2(x)
        if not self._pre_layer_norm:
            x = self.final_layer_norm(x + x_in)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_weight_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        res = x
        x_ln = (
            self.self_attention_layer_norm(x)
            if self._pre_layer_norm
            else x
        )
        out = self.self_attention(x_ln, attention_mask, attention_weight_bias)
        x = (
            res + out["embeddings"]
            if self._pre_layer_norm
            else self.self_attention_layer_norm(out["embeddings"] + res)
        )
        x = x + self.mlp(x)
        out["embeddings"] = x
        return out


class ConvBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int | None = None, kernel_size: int = 1):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, padding="same")
        self.layer_norm = LayerNormFP32(dim_in, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> LN on last dim in fp32 -> back to (B, C, L)
        x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv(x)  # compute dtype will come from the tensor
        return F.gelu(x, approximate="tanh")


class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, kernel_size: int = 1):
        super().__init__()
        self.conv_block = ConvBlock(
            dim_in=dim, dim_out=dim_out, kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ConvTowerBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # main conv (K=5) + residual conv (K=1) then downsample by 2
        self.conv = ConvBlock(dim_in=dim_in, dim_out=dim_out, kernel_size=5)
        self.res_conv = ResidualConvBlock(dim=dim_out, dim_out=dim_out, kernel_size=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.res_conv(y)
        y = self.avg_pool(y)
        return y


class DeConvUpsampleType(str, Enum):
    CONV_TRANSPOSE = "conv_transpose"
    REPEAT_CONV = "repeat+conv"


def _normalize_deconv_upsample_type(x: Any) -> DeConvUpsampleType:
    # Accept Enum, raw values, or qualified names like
    # "DeConvUpsampleType.CONV_TRANSPOSE"
    if isinstance(x, DeConvUpsampleType):
        return x
    if isinstance(x, str):
        s = x.strip()
        # strip any "Something." qualifier and normalize separators
        s_key = s.split(".")[-1].lower().replace("-", "_")
        # keep a generous alias map
        alias = {
            "conv_transpose": DeConvUpsampleType.CONV_TRANSPOSE,
            "convtranspose": DeConvUpsampleType.CONV_TRANSPOSE,
            "repeat+conv": DeConvUpsampleType.REPEAT_CONV,
            "repeat_conv": DeConvUpsampleType.REPEAT_CONV,
            "repeatconv": DeConvUpsampleType.REPEAT_CONV,
        }
        if s_key in alias:
            return alias[s_key]
        # also handle the untouched original "repeat+conv"
        if s == "repeat+conv":
            return DeConvUpsampleType.REPEAT_CONV
    raise ValueError(f"Unrecognized deconv_upsample_type: {x!r}")


class UpsamplingDeconvBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int | None = None,
        kernel_size: int = 1,
        upsample: DeConvUpsampleType | None = None,
        phase: str = "odd",
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        self.upsample = upsample
        self.kernel_size = kernel_size

        if self.upsample is None:
            self.conv = nn.ConvTranspose1d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=True,
            )
        elif self.upsample == DeConvUpsampleType.CONV_TRANSPOSE:
            self.conv = JaxConvTranspose1dSame(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=2,
                bias=True,
                phase=phase,
            )
        elif self.upsample == DeConvUpsampleType.REPEAT_CONV:
            self.conv = nn.Conv1d(
                dim_in, dim_out, kernel_size=kernel_size, padding="same"
            )
        else:
            raise ValueError(f"Invalid upsample type: {upsample}")

        self.layer_norm = LayerNormFP32(dim_in, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.upsample == DeConvUpsampleType.REPEAT_CONV:
            x = torch.repeat_interleave(x, 2, dim=-1)
        x = self.conv(x)
        return F.gelu(x, approximate="tanh")


class ResidualUpsamplingDeconvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: int = 1,
        upsample: DeConvUpsampleType | None = None,
        phase: str = "odd",
    ):
        super().__init__()
        self.conv_block = UpsamplingDeconvBlock(
            dim_in=dim,
            dim_out=dim_out,
            kernel_size=kernel_size,
            upsample=upsample,
            phase=phase,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class DeconvTowerBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        upsample: DeConvUpsampleType | None = None,
        phase: str = "odd",
    ):
        super().__init__()
        self.conv = UpsamplingDeconvBlock(
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_size=5,
            upsample=upsample,
            phase=phase,
        )
        self.res_conv = ResidualUpsamplingDeconvBlock(
            dim=dim_out, dim_out=dim_out, kernel_size=1, upsample=None, phase=phase
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.res_conv(x)
        return x


class Stem(nn.Module):
    def __init__(self, token_embed_dim: int, conv_init_embed_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(
            token_embed_dim, conv_init_embed_dim, kernel_size=15, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x), approximate="tanh")


class Core(nn.Module):
    def __init__(self, config: Ntv3PreTrainedConfig):
        super().__init__()
        self.config = config

        self._attention_layers_to_save = list(
            {t[0] for t in config.attention_maps_to_save}
        )
        self._attention_maps_per_layer_to_save = {
            layer: [t[1] for t in config.attention_maps_to_save if t[0] == layer]
            for layer in self._attention_layers_to_save
        }
        if (
            self._attention_layers_to_save
            and max(self._attention_layers_to_save) > config.num_layers
        ):
            raise ValueError("attention_maps_to_save references a non-existent layer")
        for _layer, maps in self._attention_maps_per_layer_to_save.items():
            if maps and max(maps) >= config.attention_heads:
                raise ValueError("attention map index out of range")

        self.embed_layer = nn.Embedding(config.alphabet_size, config.token_embed_dim)
        self.stem = Stem(config.token_embed_dim, config.conv_init_embed_dim)

        # Downsampling tower
        fl = copy.deepcopy(self.config.filter_list)
        self.conv_tower_blocks = nn.ModuleList(
            [
                ConvTowerBlock(dim_in=d_in, dim_out=d_out)
                for d_in, d_out in zip(fl[:-1], fl[1:])
            ]
        )

        # Transformer tower
        self.transformer_blocks = nn.ModuleList()
        rot_cfg = RotaryEmbeddingConfig(rescaling_factor=None)
        for _ in range(self.config.num_layers):
            self.transformer_blocks.append(
                SelfAttentionBlock(
                    num_heads=self.config.attention_heads,
                    embed_dim=self.config.embed_dim,
                    ffn_embed_dim=self.config.ffn_embed_dim,
                    key_size=self.config.key_size,
                    add_bias_kv=False,
                    add_bias_fnn=False,
                    ffn_activation_name="swish",
                    use_glu_in_ffn=True,
                    layer_norm_eps=self.config.layer_norm_eps,
                    pre_layer_norm=True,
                    rotary_embedding_config=rot_cfg,
                )
            )

        # Upsampling tower
        fl_rev = list(reversed(fl))
        deconv_upsample = _normalize_deconv_upsample_type(self.config.deconv_upsample_type)
        self.deconv_tower_blocks = nn.ModuleList(
            [
                DeconvTowerBlock(
                    dim_in=d_in,
                    dim_out=d_out,
                    upsample=deconv_upsample,
                    phase=self.config.deconv_phase,
                )
                for d_in, d_out in zip(fl_rev[:-1], fl_rev[1:])
            ]
        )

        # Head
        self.lm_head = nn.ModuleDict(
            {
                "hidden_layers": nn.ModuleList(
                    [
                        nn.Linear(self.config.embed_dim, self.config.embed_dim)
                        for _ in range(self.config.num_hidden_layers_head)
                    ]
                ),
                "head": nn.Linear(
                    self.config.conv_init_embed_dim, self.config.alphabet_size
                ),
            }
        )

    def conv_tower(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        residuals = []
        for block in self.conv_tower_blocks:
            residuals.append(x)
            y = block.conv(x)
            y = block.res_conv(y)
            x = block.avg_pool(y)
        return x, residuals

    def transformer_tower(
        self, x: torch.Tensor, outs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        for i, layer in enumerate(self.transformer_blocks):
            out = layer(x, attention_mask=None, attention_weight_bias=None)
            x = out["embeddings"]
            if (i + 1) in self.config.embeddings_layers_to_save:
                outs[f"embeddings_{i + 1}"] = x
            if (i + 1) in self._attention_layers_to_save:
                for m in self._attention_maps_per_layer_to_save[i + 1]:
                    outs[f"attention_map_layer_{i + 1}_number_{m}"] = out[
                        "attention_weights"
                    ][:, m + 1]
        return x, outs

    def deconv_tower(
        self,
        x: torch.Tensor,
        residuals: list[torch.Tensor],
        outs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        res_iter = reversed(residuals)
        for i, block in enumerate(self.deconv_tower_blocks):
            r = next(res_iter)
            x = block(x)
            if self.config.use_skip_connection:
                x = x + r
            if (i + 1) in self.config.deconv_layers_to_save:
                outs[f"embeddings_deconv_{i + 1}"] = x
        return x, outs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass through the Core model.

        Args:
            input_ids: Input token IDs of shape (B, L). Either input_ids or inputs_embeds must be provided.
            inputs_embeds: Pre-computed embeddings of shape (B, L, token_embed_dim). 
                           Useful for saliency/attribution analysis.
            output_hidden_states: Whether to return hidden states from each layer.
            output_attentions: Whether to return attention weights from each layer.

        Returns:
            Dictionary containing:
                - logits: Output logits of shape (B, L, alphabet_size)
                - hidden_states: (optional) List of hidden states
                - attentions: (optional) List of attention weights
                - embeddings_{i}: (optional) Specific layer embeddings if configured
                - attention_map_layer_{i}_number_{m}: (optional) Specific attention maps if configured
                - embeddings_deconv_{i}: (optional) Specific deconv layer embeddings if configured
        """
        assert (input_ids is None) != (inputs_embeds is None), \
            "You must specify exactly one of input_ids or inputs_embeds"
        
        device_type = (
            input_ids.device.type
            if input_ids is not None
            else inputs_embeds.device.type  # type: ignore
        )
        hidden_states: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        outs: dict[str, torch.Tensor | list[torch.Tensor]] = {}

        # Embedding
        if inputs_embeds is None:
            x = self.embed_layer(input_ids)  # (B, L, token_embed_dim)
        else:
            x = inputs_embeds
        emb_compute = _dtype_from_str(
            getattr(self.config, "embedding_compute_dtype", "float32")
        )
        x = x.to(emb_compute)

        # Stem
        with _autocast_to(
            device_type, getattr(self.config, "stem_compute_dtype", "bfloat16")
        ):
            x = self.stem(x.permute(0, 2, 1))  # (B, C, L)

        # Down conv tower
        with _autocast_to(
            device_type,
            getattr(self.config, "down_convolution_compute_dtype", "bfloat16"),
        ):
            residuals: list[torch.Tensor] = []
            for block in self.conv_tower_blocks:
                residuals.append(x)
                y = block.conv(x)
                y = block.res_conv(y)
                if output_hidden_states:
                    hidden_states.append(y.permute(0, 2, 1))
                x = block.avg_pool(y)

        # Transformer tower
        x = x.permute(0, 2, 1)  # (B, L, C)
        with _autocast_to(
            device_type,
            getattr(self.config, "transformer_qkvo_compute_dtype", "bfloat16"),
        ):
            for i, layer in enumerate(self.transformer_blocks):
                out = layer(x, attention_mask=None, attention_weight_bias=None)
                x = out["embeddings"]
                if output_hidden_states:
                    hidden_states.append(x)
                if output_attentions:
                    attentions.append(out["attention_weights"])
                # Save specific embeddings/attention maps if configured
                if (i + 1) in self.config.embeddings_layers_to_save:
                    outs[f"embeddings_{i + 1}"] = x
                if (i + 1) in self._attention_layers_to_save:
                    for m in self._attention_maps_per_layer_to_save[i + 1]:
                        outs[f"attention_map_layer_{i + 1}_number_{m}"] = out[
                            "attention_weights"
                        ][:, m + 1]

        # Deconv tower
        x = x.permute(0, 2, 1)  # (B, C, L)
        with _autocast_to(
            device_type,
            getattr(self.config, "up_convolution_compute_dtype", "bfloat16"),
        ):
            for i, block in enumerate(self.deconv_tower_blocks):
                r = residuals[-(i + 1)]
                y = block(x)
                if self.config.use_skip_connection:
                    y = y + r
                x = y
                if output_hidden_states:
                    hidden_states.append(x.permute(0, 2, 1))
                if (i + 1) in self.config.deconv_layers_to_save:
                    outs[f"embeddings_deconv_{i + 1}"] = x

        # Head
        y = x.permute(0, 2, 1)  # (B, L, C)
        with _autocast_to(
            device_type, getattr(self.config, "lmhead_compute_dtype", "float32")
        ):
            y = F.gelu(y, approximate="tanh")
            for hl in self.lm_head["hidden_layers"]:
                y = F.gelu(hl(y), approximate="tanh")
            # Ensure matmul dtypes match the linear weight dtype
            y = y.to(self.lm_head["head"].weight.dtype)
            logits = self.lm_head["head"](y)

        outs["logits"] = logits
        if output_hidden_states:
            outs["hidden_states"] = hidden_states
        if output_attentions:
            outs["attentions"] = attentions

        return outs


class NTv3PreTrained(PreTrainedModel):
    config_class = Ntv3PreTrainedConfig
    base_model_prefix = "ntv3"

    def __init__(self, config: Ntv3PreTrainedConfig):
        super().__init__(config)
        self.core = Core(config)
        self.post_init()

    def tie_weights(self, **kwargs) -> None:  # input/output embeddings are different dims
        return

    def get_input_embeddings(self) -> nn.Embedding:
        return self.core.embed_layer

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.core.embed_layer = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.core.lm_head["head"]

    def set_output_embeddings(self, new_lm_head: nn.Module) -> None:
        self.core.lm_head["head"] = new_lm_head

    def resize_token_embeddings(
        self, new_num_tokens: int | None = None
    ) -> nn.Embedding:
        old = self.get_input_embeddings()
        old_n, dim = old.weight.shape
        if not new_num_tokens or new_num_tokens == old_n:
            return old

        new_emb = nn.Embedding(
            new_num_tokens, dim, device=old.weight.device, dtype=old.weight.dtype
        )
        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
        n = min(old_n, new_num_tokens)
        new_emb.weight.data[:n] = old.weight.data[:n]
        self.set_input_embeddings(new_emb)

        out = self.get_output_embeddings()
        new_out = nn.Linear(
            out.in_features,
            new_num_tokens,
            bias=True,
            device=out.weight.device,
            dtype=out.weight.dtype,
        )
        nn.init.normal_(new_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(new_out.bias)
        new_out.weight.data[:n] = out.weight.data[:n]
        new_out.bias.data[:n] = out.bias.data[:n]
        self.set_output_embeddings(new_out)

        self.config.alphabet_size = new_num_tokens
        return new_emb

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: dict[str, Any],
    ) -> MaskedLMOutput | tuple:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        collect_h = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        collect_a = (
            output_attentions
            if output_attentions is not None
            else getattr(self.config, "output_attentions", False)
        )

        # Forward through core
        outs = self.core(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=collect_h,
            output_attentions=collect_a,
        )

        logits = outs["logits"]
        hidden_states = outs.get("hidden_states")
        attentions = outs.get("attentions")

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        if not return_dict:
            out = (logits,)
            if hidden_states:
                out += (tuple(hidden_states),)
            if attentions:
                out += (tuple(attentions),)
            return out

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(hidden_states) if hidden_states else None,
            attentions=tuple(attentions) if attentions else None,
        )


__all__ = [
    "NTv3PreTrained",
    "Core",
    "ConvBlock",
    "ConvTowerBlock",
    "DeconvTowerBlock",
    "DeConvUpsampleType",
    "RotaryEmbeddingConfig",
    "SelfAttentionBlock",
    "Stem",
    "UpsamplingDeconvBlock",
    "_autocast_to",
    "_normalize_deconv_upsample_type",
]

