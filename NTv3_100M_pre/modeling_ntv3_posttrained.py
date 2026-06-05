from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput

try:
    from .configuration_ntv3_posttrained import (
        DiscreteConditionedNTv3Config,
        NTv3PostTrainedConfig,
    )
except ImportError:
    from configuration_ntv3_posttrained import (  # type: ignore
        DiscreteConditionedNTv3Config,
        NTv3PostTrainedConfig,
    )

try:
    from .configuration_ntv3_pretrained import Ntv3PreTrainedConfig
except ImportError:
    from configuration_ntv3_pretrained import Ntv3PreTrainedConfig  # type: ignore

try:
    from .modeling_ntv3_pretrained import (
        ConvBlock,
        DeConvUpsampleType,
        LayerNormFP32,
        RotaryEmbeddingConfig,
        SelfAttentionBlock,
        UpsamplingDeconvBlock,
        Core as NTv3PreTrainedCore,
        _autocast_to,
        _normalize_deconv_upsample_type,
    )
except ImportError:
    from modeling_ntv3_pretrained import (  # type: ignore
        ConvBlock,
        DeConvUpsampleType,
        LayerNormFP32,
        RotaryEmbeddingConfig,
        SelfAttentionBlock,
        UpsamplingDeconvBlock,
        Core as NTv3PreTrainedCore,
        _autocast_to,
        _normalize_deconv_upsample_type,
    )

__all__ = [
    "NTv3PreTrainedCore",
    "ConditionedNTv3PreTrainedCore",
    "DiscreteConditionedNTv3PreTrainedCore",
    "NTv3PostTrainedCore",
    "NTv3PostTrained",
    "NTv3Generative"
    "NTv3PostTrainedOutput",
]


@dataclass
class NTv3PostTrainedOutput(ModelOutput):
    """
    Output class for NTv3 post-trained model.

    Args:
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        bigwig_tracks_logits (`torch.Tensor`, *optional*):
            BigWig track predictions of shape `(batch_size, sequence_length, num_tracks)`.
            Only present if the model has a bigwig head configured.
        bed_tracks_logits (`torch.Tensor`, *optional*):
            BED element predictions of shape `(batch_size, sequence_length, num_elements, num_classes)`.
            Only present if the model has a bed head configured.
        embedding (`torch.Tensor` of shape `(batch_size, sequence_length, config.embed_dim)`):
            Intermediate embedding after deconv tower and before heads.
        after_transformer_embedding (`torch.Tensor` of shape `(batch_size, sequence_length, config.embed_dim)`):
            Embedding after transformer tower.
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax.
    """

    logits: Optional[torch.Tensor] = None
    bigwig_tracks_logits: Optional[torch.Tensor] = None
    bed_tracks_logits: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None
    after_transformer_embedding: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None


class AdaptiveLayerNorm(nn.LayerNorm):
    """LayerNorm that applies per-condition affine modulation."""

    def __init__(
        self, num_features: int, conditions_dims: list[int], epsilon: float = 1e-5
    ):
        super().__init__(
            normalized_shape=num_features, eps=epsilon, elementwise_affine=True
        )
        self.modulation_layers = nn.ModuleList(
            [nn.Linear(cd, 2 * num_features) for cd in conditions_dims]
        )
        self._num_conditions = len(conditions_dims)
        self._dim = num_features

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self._base_ln_fp32(x)

        if len(conditions) != self._num_conditions:
            raise ValueError("Number of conditions mismatch")

        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]

        scale = torch.ones_like(x[:, :1, :])
        shift = torch.zeros_like(x[:, :1, :])

        for i, cond in enumerate(conditions):
            cond_cast = cond.to(self.modulation_layers[i].weight.dtype)
            tmp = self.modulation_layers[i](cond_cast).unsqueeze(1)
            tmp = tmp.to(x.dtype)
            shift_i, scale_i = torch.chunk(tmp, 2, dim=-1)
            mask = conditions_masks[i].unsqueeze(-1).unsqueeze(-1)
            shift_i = torch.where(mask.bool(), shift_i, 0.0)
            scale_i = torch.where(mask.bool(), scale_i, 0.0)
            scale = scale * (1.0 + scale_i)
            shift = shift + shift_i

        return x * scale + shift

    def _base_ln_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """Run base LayerNorm in fp32 (compiler-friendly, like Mistral/Gemma)."""
        # Compute in fp32
        x_fp32 = x.to(torch.float32)
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
        x_normed = (x_fp32 - mean) * torch.rsqrt(var + self.eps)
        
        # Apply inherited weight/bias in fp32, then cast back
        if self.weight is not None:
            x_normed = x_normed * self.weight.to(torch.float32)
        if self.bias is not None:
            x_normed = x_normed + self.bias.to(torch.float32)
        
        return x_normed.type_as(x)


class AdaptiveConvBlock(ConvBlock):
    """Convolutional block with condition-aware layer normalisation."""

    def __init__(
        self,
        dim: int,
        conditions_dims: list[int],
        dim_out: int | None = None,
        kernel_size: int = 1,
    ):
        super().__init__(dim_in=dim, dim_out=dim_out, kernel_size=kernel_size)
        self.layer_norm = AdaptiveLayerNorm(dim, conditions_dims)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x, conditions, conditions_masks)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return F.gelu(x, approximate="tanh")


class AdaptiveResidualConvBlock(nn.Module):
    """Residual conv block gated by condition-specific scalars."""

    def __init__(
        self,
        dim: int,
        conditions_dims: list[int],
        dim_out: int | None = None,
        kernel_size: int = 1,
    ):
        super().__init__()
        self.conv_block = AdaptiveConvBlock(
            dim, conditions_dims, dim_out=dim_out, kernel_size=kernel_size
        )
        self.modulation_layers = nn.ModuleList(
            [nn.Linear(cd, dim) for cd in conditions_dims]
        )

    def forward(
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]
        gate = 1.0
        for i, cond in enumerate(conditions):
            cond_cast = cond.to(self.modulation_layers[i].weight.dtype)
            g = self.modulation_layers[i](cond_cast).unsqueeze(-1)
            g = g.to(x.dtype)
            mask = conditions_masks[i].unsqueeze(-1).unsqueeze(-1)
            g = torch.where(mask.bool(), g, 0.0)
            gate = gate * (1.0 + g)
        return x + gate * self.conv_block(x, conditions, conditions_masks)


class AdaptiveDeConvBlock(UpsamplingDeconvBlock):
    """Upsampling block with adaptive normalisation."""

    def __init__(
        self,
        dim: int,
        conditions_dims: list[int],
        dim_out: int | None = None,
        kernel_size: int = 1,
        upsample: DeConvUpsampleType | None = None,
        phase: str = "odd",
    ):
        super().__init__(
            dim_in=dim,
            dim_out=dim_out,
            kernel_size=kernel_size,
            upsample=upsample,
            phase=phase,
        )
        self.layer_norm = AdaptiveLayerNorm(dim, conditions_dims)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]
        x = self.layer_norm(x.permute(0, 2, 1), conditions, conditions_masks).permute(
            0, 2, 1
        )
        if self.upsample == DeConvUpsampleType.REPEAT_CONV:
            x = torch.repeat_interleave(x, 2, dim=-1)
        x = self.conv(x)
        return F.gelu(x, approximate="tanh")


class AdaptiveResidualDeConvBlock(nn.Module):
    """Residual deconv block gated by condition-specific scalars."""

    def __init__(
        self,
        dim: int,
        conditions_dims: list[int],
        dim_out: int | None = None,
        kernel_size: int = 1,
        upsample: DeConvUpsampleType | None = None,
    ):
        super().__init__()
        self.conv_block = AdaptiveDeConvBlock(
            dim,
            conditions_dims,
            dim_out=dim_out,
            kernel_size=kernel_size,
            upsample=upsample,
        )
        self.modulation_layers = nn.ModuleList(
            [nn.Linear(cd, dim) for cd in conditions_dims]
        )

    def forward(
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]
        gate = 1.0
        for i, cond in enumerate(conditions):
            cond_cast = cond.to(self.modulation_layers[i].weight.dtype)
            g = self.modulation_layers[i](cond_cast).unsqueeze(-1)
            g = g.to(x.dtype)
            mask = conditions_masks[i].unsqueeze(-1).unsqueeze(-1)
            g = torch.where(mask.bool(), g, 0.0)
            gate = gate * (1.0 + g)
        return x + gate * self.conv_block(x, conditions, conditions_masks)


class AdaptiveSelfAttentionBlock(SelfAttentionBlock):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        conditions_dims: list[int],
        key_size: int | None = None,
        add_bias_kv: bool = False,
        add_bias_fnn: bool = False,
        use_glu_in_ffn: bool = True,
        layer_norm_eps: float = 1e-5,
        pre_layer_norm: bool = True,
        rotary_embedding_config: RotaryEmbeddingConfig | None = None,
    ):
        super().__init__(
            num_heads=num_heads,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            key_size=key_size,
            add_bias_kv=add_bias_kv,
            add_bias_fnn=add_bias_fnn,
            ffn_activation_name="swish",
            use_glu_in_ffn=use_glu_in_ffn,
            layer_norm_eps=layer_norm_eps,
            pre_layer_norm=pre_layer_norm,
            rotary_embedding_config=rotary_embedding_config,
        )
        self.self_attention_layer_norm = AdaptiveLayerNorm(
            embed_dim, conditions_dims, epsilon=layer_norm_eps
        )
        self.final_layer_norm = AdaptiveLayerNorm(
            embed_dim, conditions_dims, epsilon=layer_norm_eps
        )
        self.modulation_layers = nn.ModuleList(
            [nn.Linear(cd, embed_dim) for cd in conditions_dims]
        )

    def mlp(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        if self._pre_layer_norm:
            x_norm = self.final_layer_norm(x, conditions, conditions_masks)
        else:
            x_norm = x
        if self._use_glu_in_fnn:
            x_lin = self.fc1(x_norm)
            x1, x2 = torch.chunk(x_lin, 2, dim=-1)
            x_mlp = self._ffn_activation_fn(x1) * x2
        else:
            x_mlp = self._ffn_activation_fn(self.fc1(x_norm))
        x_mlp = self.fc2(x_mlp)
        if not self._pre_layer_norm:
            x_mlp = self.final_layer_norm(x + x_mlp, conditions, conditions_masks)
        return x_mlp

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_weight_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if conditions_masks is None:
            conditions_masks = [
                torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
                for _ in conditions
            ]

        res = x
        if self._pre_layer_norm:
            x_norm = self.self_attention_layer_norm(x, conditions, conditions_masks)
        else:
            x_norm = x
        out = self.self_attention(x_norm, attention_mask, attention_weight_bias)
        attn_out, attn_weights = out["embeddings"], out["attention_weights"]
        if self._pre_layer_norm:
            x = res + attn_out
        else:
            x = self.self_attention_layer_norm(
                res + attn_out, conditions, conditions_masks
            )
        res = x
        if self._pre_layer_norm:
            gate = 1.0
            for i, cond in enumerate(conditions):
                cond_cast = cond.to(self.modulation_layers[i].weight.dtype)
                g = self.modulation_layers[i](cond_cast).unsqueeze(1)
                g = g.to(x.dtype)
                mask = conditions_masks[i].unsqueeze(-1).unsqueeze(-1)
                g = torch.where(mask.bool(), g, 0.0)
                gate = gate * (1.0 + g)
            x = res + gate * self.mlp(x, conditions, conditions_masks)
        else:
            mlp_out = self.mlp(x, conditions, conditions_masks)
            x = self.final_layer_norm(res + mlp_out, conditions, conditions_masks)
        return {"embeddings": x, "attention_weights": attn_weights}


class ConditionedConvTowerBlock(nn.Module):
    """Condition-aware variant of the convolutional tower block."""

    def __init__(self, dim_in: int, dim_out: int, conditions_dims: list[int]):
        super().__init__()
        self.conv = AdaptiveConvBlock(
            dim_in, conditions_dims, dim_out=dim_out, kernel_size=5
        )
        self.res_conv = AdaptiveResidualConvBlock(
            dim_out, conditions_dims, dim_out=dim_out, kernel_size=1
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        y = self.conv(x, conditions, conditions_masks)
        y = self.res_conv(y, conditions, conditions_masks)
        return self.avg_pool(y)


class ConditionedDeConvTowerBlock(nn.Module):
    """Condition-aware variant of the upsampling tower block."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        conditions_dims: list[int],
        upsample_type: DeConvUpsampleType,
        phase: str = "odd",
    ):
        super().__init__()
        self.conv = AdaptiveDeConvBlock(
            dim_in,
            conditions_dims,
            dim_out=dim_out,
            kernel_size=5,
            upsample=upsample_type,
            phase=phase,
        )
        self.res_conv = AdaptiveResidualDeConvBlock(
            dim_out, conditions_dims, dim_out=dim_out, kernel_size=1, upsample=None
        )

    def forward(
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        y = self.conv(x, conditions, conditions_masks)
        return self.res_conv(y, conditions, conditions_masks)


class LinearHead(nn.Module):
    """A linear head that predicts one scalar value per track."""

    def __init__(self, embed_dim: int, num_labels: int):
        super().__init__()
        self.layer_norm = LayerNormFP32(embed_dim)
        self.head = nn.Linear(in_features=embed_dim, out_features=num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = self.layer_norm(x.float())
        x = self.head(x)
        x = F.softplus(x)
        return x.to(orig_dtype)


class ZeroHead(nn.Module):
    """Placeholder head for assemblies with no associated tracks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((*x.shape[:-1], 0))


class MultiSpeciesHead(nn.Module):
    """Predict sets of tracks for each species-specific head."""

    def __init__(
        self,
        num_tracks_per_head: list[int],
        embed_dim: int,
    ):
        super().__init__()
        self.num_output_tracks = max(num_tracks_per_head)
        self.species_heads = nn.ModuleList(
            [
                (
                    LinearHead(embed_dim=embed_dim, num_labels=num_tracks_per_head_i)
                    if num_tracks_per_head_i > 0
                    else ZeroHead()
                )
                for num_tracks_per_head_i in num_tracks_per_head
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        head_idx: torch.Tensor | int,
        output_track: bool = True,
    ) -> torch.Tensor:
        if torch.is_tensor(head_idx):
            head_idx_tensor = head_idx.to(x.device)
            if head_idx_tensor.ndim == 0:
                head_idx_tensor = head_idx_tensor.expand(x.size(0))
            elif head_idx_tensor.ndim == 1:
                if head_idx_tensor.numel() == 1:
                    head_idx_tensor = head_idx_tensor.expand(x.size(0))
                elif head_idx_tensor.numel() != x.size(0):
                    raise ValueError("head_idx tensor must broadcast to batch size")
            else:
                raise ValueError("head_idx tensor must be 0D or 1D")
        else:
            head_idx_tensor = torch.full(
                (x.size(0),), int(head_idx), dtype=torch.long, device=x.device
            )

        logits = x.new_zeros((*x.shape[:-1], self.num_output_tracks))
        for i, head in enumerate(self.species_heads):
            head_logits = head(x)
            if output_track:
                if torch.all(head_idx_tensor == i):
                    return head_logits
            head_padding = self.num_output_tracks - head_logits.shape[-1]
            if head_padding > 0:
                padded_logits = F.pad(head_logits, (0, head_padding))
            else:
                padded_logits = head_logits

            mask_b = head_idx_tensor == i
            mask = mask_b.view(-1, 1, 1)
            logits = torch.where(mask, padded_logits, logits)

        return logits


class ClassificationHead(nn.Module):
    """A linear head that predicts num_classes values per label."""

    def __init__(self, embed_dim: int, num_labels: int, num_classes: int):
        super().__init__()
        self.layer_norm = LayerNormFP32(embed_dim)
        self.head = nn.Linear(
            in_features=embed_dim, out_features=num_labels * num_classes
        )
        self.num_labels = num_labels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = self.layer_norm(x.float())
        x = self.head(x)
        x = torch.reshape(x, x.shape[:-1] + (self.num_labels, self.num_classes))
        return x.to(orig_dtype)


class ConditionedNTv3PreTrainedCore(NTv3PreTrainedCore):
    """
    NTv3 with condition-aware adaptive layers.

    Replaces regular towers with adaptive versions that modulate based on
    condition embeddings. Takes condition embeddings as input (not token IDs).
    """

    def __init__(self, config: Ntv3PreTrainedConfig, conditions_dims: list[int]):
        super().__init__(config)
        self.conditions_dims = conditions_dims

        # Replace conv tower with adaptive version
        fl = copy.deepcopy(config.filter_list)
        self.conv_tower_blocks = nn.ModuleList(
            [
                ConditionedConvTowerBlock(d_in, d_out, conditions_dims)
                for d_in, d_out in zip(fl[:-1], fl[1:])
            ]
        )

        # Replace transformer tower with adaptive version
        rot_cfg = RotaryEmbeddingConfig(rescaling_factor=None)
        self.transformer_blocks = nn.ModuleList(
            [
                AdaptiveSelfAttentionBlock(
                    config.attention_heads,
                    config.embed_dim,
                    config.ffn_embed_dim,
                    conditions_dims,
                    config.key_size,
                    rotary_embedding_config=rot_cfg,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Replace deconv tower with adaptive version
        fl_rev = list(reversed(fl))
        deconv_upsample = _normalize_deconv_upsample_type(config.deconv_upsample_type)
        self.deconv_tower_blocks = nn.ModuleList(
            [
                ConditionedDeConvTowerBlock(
                    d_in, d_out, conditions_dims, deconv_upsample, config.deconv_phase
                )
                for d_in, d_out in zip(fl_rev[:-1], fl_rev[1:])
            ]
        )

    def conv_tower(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
        """Adaptive downsampling convolutional tower."""
        residuals: list[torch.Tensor] = []
        hidden_states: list[torch.Tensor] = [] if output_hidden_states else None
        for block in self.conv_tower_blocks:
            residuals.append(x)
            y = block.conv(x, conditions, conditions_masks)
            y = block.res_conv(y, conditions, conditions_masks)
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(y.permute(0, 2, 1))
            x = block.avg_pool(y)
        return x, residuals, hidden_states

    def transformer_tower(  # type: ignore[override]
        self,
        x: torch.Tensor,
        outs: dict[str, torch.Tensor],
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Adaptive transformer tower."""
        hidden_states: list[torch.Tensor] = [] if output_hidden_states else None
        attentions: list[torch.Tensor] = [] if output_attentions else None
        for i, layer in enumerate(self.transformer_blocks):
            out = layer(
                x,
                conditions,
                conditions_masks,
                attention_mask=None,
                attention_weight_bias=None,
            )
            x = out["embeddings"]
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(x)
            if output_attentions and attentions is not None:
                attentions.append(out["attention_weights"])
            if (i + 1) in self.config.embeddings_layers_to_save:
                outs[f"embeddings_{i + 1}"] = x
            if (i + 1) in self._attention_layers_to_save:
                for m in self._attention_maps_per_layer_to_save[i + 1]:
                    outs[f"attention_map_layer_{i + 1}_number_{m}"] = out[
                        "attention_weights"
                    ][:, m + 1]
        return x, outs, hidden_states, attentions

    def deconv_tower(  # type: ignore[override]
        self,
        x: torch.Tensor,
        residuals: list[torch.Tensor],
        outs: dict[str, torch.Tensor],
        conditions: list[torch.Tensor],
        conditions_masks: list[torch.Tensor] | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor] | None]:
        """Adaptive upsampling deconvolutional tower."""
        hidden_states: list[torch.Tensor] = [] if output_hidden_states else None
        for i, block in enumerate(self.deconv_tower_blocks):
            r = residuals[-(i + 1)]
            y = block.conv(x, conditions, conditions_masks)
            y = block.res_conv(y, conditions, conditions_masks)
            if self.config.use_skip_connection:
                y = y + r
            x = y
            if output_hidden_states and hidden_states is not None:
                hidden_states.append(x.permute(0, 2, 1))
            if (i + 1) in self.config.deconv_layers_to_save:
                outs[f"embeddings_deconv_{i + 1}"] = x
        return x, outs, hidden_states

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        conditions: list[torch.Tensor] | None = None,
        conditions_masks: list[torch.Tensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass with condition embeddings.

        Args:
            input_ids: Input token IDs of shape (B, L). Either input_ids or inputs_embeds must be provided.
            conditions: List of condition embeddings, each of shape (B, condition_dim).
            conditions_masks: Optional masks for each condition.
            inputs_embeds: Pre-computed embeddings of shape (B, L, token_embed_dim).
                           Useful for saliency/attribution analysis.
            output_hidden_states: Whether to return hidden states from each layer.
            output_attentions: Whether to return attention weights from each layer.

        Returns:
            Dictionary containing logits, intermediate embeddings, hidden_states, and attentions.
        """
        assert (input_ids is None) != (inputs_embeds is None), \
            "You must specify exactly one of input_ids or inputs_embeds"
        
        device_type = (
            input_ids.device.type
            if input_ids is not None
            else inputs_embeds.device.type  # type: ignore
        )
        outs: dict[str, torch.Tensor | list[torch.Tensor]] = {}

        # Embedding
        with _autocast_to(device_type, self.config.embedding_compute_dtype):
            if inputs_embeds is None:
                x = self.embed_layer(input_ids)
            else:
                x = inputs_embeds

        # Stem
        with _autocast_to(device_type, self.config.stem_compute_dtype):
            x = self.stem(x.permute(0, 2, 1))

        # Conv tower (adaptive)
        with _autocast_to(device_type, self.config.down_convolution_compute_dtype):
            x, residuals, conv_hidden_states = self.conv_tower(
                x, conditions, conditions_masks, output_hidden_states=output_hidden_states
            )

        # Transformer tower (adaptive)
        x = x.permute(0, 2, 1)
        with _autocast_to(device_type, self.config.transformer_qkvo_compute_dtype):
            x, outs, transformer_hidden_states, attentions = self.transformer_tower(
                x, outs, conditions, conditions_masks,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )
        outs["after_transformer_embedding"] = x

        # Deconv tower (adaptive)
        x = x.permute(0, 2, 1)
        with _autocast_to(device_type, self.config.up_convolution_compute_dtype):
            x, outs, deconv_hidden_states = self.deconv_tower(
                x, residuals, outs, conditions, conditions_masks,
                output_hidden_states=output_hidden_states
            )

        # LM Head
        y = x.permute(0, 2, 1)
        outs["embedding"] = y
        with _autocast_to(device_type, self.config.lmhead_compute_dtype):
            logits = F.gelu(y, approximate="tanh")
            for hl in self.lm_head["hidden_layers"]:
                logits = F.gelu(hl(logits), approximate="tanh")
            logits = logits.to(self.lm_head["head"].weight.dtype)
            logits = self.lm_head["head"](logits)
        outs["logits"] = logits

        # Combine hidden states from all towers
        if output_hidden_states:
            hidden_states = []
            if conv_hidden_states:
                hidden_states.extend(conv_hidden_states)
            if transformer_hidden_states:
                hidden_states.extend(transformer_hidden_states)
            if deconv_hidden_states:
                hidden_states.extend(deconv_hidden_states)
            outs["hidden_states"] = hidden_states

        if output_attentions and attentions:
            outs["attentions"] = attentions

        return outs


class DiscreteConditionedNTv3PreTrainedCore(ConditionedNTv3PreTrainedCore):
    """
    NTv3 conditioned on discrete tokens.

    Adds embedding tables to convert discrete condition token IDs to embeddings,
    and condition prediction heads for joint modeling.
    """

    def __init__(self, config: DiscreteConditionedNTv3Config):
        super().__init__(
            config=config, 
            conditions_dims=[
                config.token_embed_dim for _ in config.conditions_vocab_size
            ],
        )

        self.cond_tables = nn.ModuleList(
            [
                nn.Embedding(v, self.config.token_embed_dim)
                for v in config.conditions_vocab_size
            ]
        )

        # Condition prediction heads (for joint modeling)
        self.conditions_heads = nn.ModuleList(
            [nn.Linear(config.embed_dim, v) for v in config.conditions_vocab_size]
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        condition_ids: list[torch.Tensor] | None = None,
        conditions_masks: list[torch.Tensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass with discrete condition tokens.

        Args:
            input_ids: Input token IDs of shape (B, L). Either input_ids or inputs_embeds must be provided.
            condition_ids: List of condition token IDs, each of shape (B,).
            conditions_masks: Optional masks for each condition.
            inputs_embeds: Pre-computed embeddings of shape (B, L, token_embed_dim).
                           Useful for saliency/attribution analysis.
            output_hidden_states: Whether to return hidden states from each layer.
            output_attentions: Whether to return attention weights from each layer.

        Returns:
            Dictionary containing logits, condition logits, and intermediate embeddings.
        """
        assert (input_ids is None) != (inputs_embeds is None), \
            "You must specify exactly one of input_ids or inputs_embeds"
        
        # Get batch size from either input_ids or inputs_embeds
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]  # type: ignore
        assert all(
            cond.shape == (batch_size,) for cond in condition_ids  # type: ignore
        ), "condition_ids must be shape (B, ) for each condition"

        device_type = (
            input_ids.device.type
            if input_ids is not None
            else inputs_embeds.device.type  # type: ignore
        )

        # Encode conditions
        with _autocast_to(device_type, self.config.modulation_compute_dtype):
            conditions = [emb(ids) for emb, ids in zip(self.cond_tables, condition_ids)]  # type: ignore

        # Call parent forward with condition embeddings
        outs = super().forward(
            input_ids=input_ids, 
            conditions=conditions, 
            conditions_masks=conditions_masks,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        # Add condition prediction heads
        for i, head in enumerate(self.conditions_heads):
            embed_mean = outs["after_transformer_embedding"].mean(dim=1)
            cond_logits = head(embed_mean.to(head.weight.dtype))
            outs[f"condition_{i}_logits"] = cond_logits

        return outs


class NTv3PostTrainedCore(DiscreteConditionedNTv3PreTrainedCore):
    """
    Full NTv3 post-trained model core with bigwig, bed and MLM head.
    """

    def __init__(self, config: NTv3PostTrainedConfig):
        super().__init__(config)

        # BigWig head
        if len(config.bigwigs_per_species) > 0:
            # we take the convention that the order of the species heads is the order 
            # of the species tokens ids
            sorted_species = sorted(
                config.bigwigs_per_species.keys(),
                key=lambda s: config.species_to_token_id[s],
            )
            sorted_num_tracks_per_head = [
                len(config.bigwigs_per_species[s]) for s in sorted_species
            ]
            self.bigwig_head: MultiSpeciesHead | None = MultiSpeciesHead(
                num_tracks_per_head=sorted_num_tracks_per_head,
                embed_dim=config.embed_dim,
            )
        else:
            self.bigwig_head = None

        # BED head
        if len(config.bed_elements_names) > 0:
            self.bed_head: ClassificationHead | None = ClassificationHead(
                embed_dim=config.embed_dim,
                num_labels=len(config.bed_elements_names),
                num_classes=2,
            )
        else:
            self.bed_head = None

    def _crop_to_center(self, x: torch.Tensor) -> torch.Tensor:
        """Crop sequence to center based on keep_target_center_fraction."""
        keep_frac = self.config.keep_target_center_fraction
        if keep_frac < 1.0:
            seq_len = x.shape[1]
            crop_len = int(seq_len * keep_frac)
            start = (seq_len - crop_len) // 2
            return x[:, start : start + crop_len, :]
        return x

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        species_ids: torch.Tensor | None = None,
        output_track: bool = False,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass with species IDs.

        Args:
            input_ids: Input token IDs of shape (B, L). Either input_ids or inputs_embeds must be provided.
            species_ids: Species token IDs of shape (B,).
            output_track: Whether to output track predictions.
            inputs_embeds: Pre-computed embeddings of shape (B, L, token_embed_dim).
                           Useful for saliency/attribution analysis.
            output_hidden_states: Whether to return hidden states from each layer.
            output_attentions: Whether to return attention weights from each layer.

        Returns:
            Dictionary containing logits, bigwig_tracks_logits, bed_tracks_logits, etc.
        """
        assert (input_ids is None) != (inputs_embeds is None), \
            "You must specify exactly one of input_ids or inputs_embeds"
        
        # Get batch size from either input_ids or inputs_embeds
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]  # type: ignore
        
        assert species_ids is not None, "species_ids must be provided"
        assert species_ids.shape == (batch_size,), \
            f"species_ids must be shape (B,), got {species_ids.shape}"

        assert (
            species_ids.min() >= self.config.num_species_special_tokens
        ), (
            f"invalid species_ids: species_ids must be greater than "
            f"or equal to {self.config.num_species_special_tokens}"
        )

        assert (
            species_ids.max()
            < self.config.num_species_special_tokens + len(self.config.bigwigs_per_species)
        ), (
            f"invalid species_ids: species_ids must be less than "
            f"{self.config.num_species_special_tokens + len(self.config.bigwigs_per_species)}"
        )

        # Compute head_idx internally. Species token id includes special tokens, 
        # however we do not have a head for the special tokens, so we subtract the 
        # number of special tokens to get the head index.
        head_idx = species_ids - self.config.num_species_special_tokens

        outs = super().forward(
            input_ids=input_ids,
            condition_ids=[species_ids],
            conditions_masks=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        embedding = self._crop_to_center(outs["embedding"])

        # BigWig head
        if self.bigwig_head is not None:
            bigwig_logits = self.bigwig_head(
                embedding, head_idx, output_track=output_track
            )
            outs["bigwig_tracks_logits"] = bigwig_logits

        # BED head
        if self.bed_head is not None:
            bed_logits = self.bed_head(embedding)
            outs["bed_tracks_logits"] = bed_logits

        return outs


class NTv3PostTrained(PreTrainedModel):
    """
    HuggingFace PreTrainedModel wrapper for post-trained NTv3.

    Provides a clean user-facing API with species_ids input.
    Returns multiple outputs (logits, tracks, elements) - loss should be
    computed externally by training code.
    """

    config_class = NTv3PostTrainedConfig
    base_model_prefix = "ntv3"

    def __init__(self, config: NTv3PostTrainedConfig):
        super().__init__(config)
        self.core = NTv3PostTrainedCore(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        species_ids: torch.Tensor | None = None,
        output_track: bool = False,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: dict[str, Any],
    ) -> NTv3PostTrainedOutput | tuple:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (B, L). Either input_ids or inputs_embeds must be provided.
            species_ids: Species token IDs of shape (B,).
            output_track: Whether to output track predictions.
            inputs_embeds: Pre-computed embeddings of shape (B, L, token_embed_dim).
                           Useful for saliency/attribution analysis.
            output_hidden_states: Whether to return hidden states from each layer.
            output_attentions: Whether to return attention weights from each layer.
            return_dict: Whether to return a ModelOutput instead of a tuple.

        Returns:
            If `return_dict=False`, returns a tuple containing:
                - logits
                - bigwig_tracks_logits (if present)
                - bed_tracks_logits (if present)
                - hidden_states (if output_hidden_states=True)
                - attentions (if output_attentions=True)
            Otherwise returns NTv3PostTrainedOutput.
        """
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
        outs = self.core(
            input_ids=input_ids,
            species_ids=species_ids,
            output_track=output_track,
            inputs_embeds=inputs_embeds,
            output_hidden_states=collect_h,
            output_attentions=collect_a,
        )

        logits = outs.get("logits")
        bigwig_tracks_logits = outs.get("bigwig_tracks_logits")
        bed_tracks_logits = outs.get("bed_tracks_logits")
        hidden_states = outs.get("hidden_states")
        attentions = outs.get("attentions")

        if not return_dict:
            out = (logits,)
            if bigwig_tracks_logits is not None:
                out += (bigwig_tracks_logits,)
            if bed_tracks_logits is not None:
                out += (bed_tracks_logits,)
            if hidden_states:
                out += (tuple(hidden_states),)
            if attentions:
                out += (tuple(attentions),)
            return out  # type: ignore

        return NTv3PostTrainedOutput(
            logits=logits,
            bigwig_tracks_logits=bigwig_tracks_logits,
            bed_tracks_logits=bed_tracks_logits,
            embedding=outs.get("embedding"),
            after_transformer_embedding=outs.get("after_transformer_embedding"),
            hidden_states=tuple(hidden_states) if hidden_states else None,
            attentions=tuple(attentions) if attentions else None,
        )
    
    @property
    def supported_species(self) -> list[str]:
        """List of supported species names (excludes special tokens)."""
        return sorted([k for k in self.config.species_to_token_id.keys() if not k.startswith("<")])

    def encode_species(self, species: str | list[str]) -> torch.LongTensor:
        """
        Encode species name(s) to token IDs.

        Args:
            species: Species name(s) (e.g., "human" or ["human", "mouse"]).
                Use `model.supported_species` to see all valid options.

        Returns:
            Tensor of shape (len(species),) containing token IDs,
            on the same device as the model.

        Raises:
            ValueError: If any species name is not supported.

        Example:
            >>> print(model.supported_species)  # See valid species
            >>> species_ids = model.encode_species(["human", "mouse"])
            >>> out = model(input_ids=tokens, species_ids=species_ids)
        """
        if isinstance(species, str):
            species = [species]
        token_ids = []
        for s in species:
            if s not in self.supported_species:
                raise ValueError(
                    f"Unknown species '{s}'. Supported species: {self.supported_species}"
                )
            token_ids.append(self.config.species_to_token_id[s])
        return torch.LongTensor(token_ids)


class NTv3Generative(PreTrainedModel):
    """
    HuggingFace PreTrainedModel wrapper for NTv3-generative.

    Provides a clean user-facing API with species_ids input.
    Returns multiple outputs (logits for sequences and conditions, elements) - loss should be
    computed externally by training code.
    """
    
    config_class = DiscreteConditionedNTv3Config
    base_model_prefix = "ntv3"
    
    def __init__(self, config: NTv3PostTrainedConfig):
        super().__init__(config)
        self.core = DiscreteConditionedNTv3PreTrainedCore(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        condition_ids: list[torch.Tensor] | None = None,
        conditions_masks: list[torch.Tensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass.
        Args:
            input_ids: Input token IDs of shape (B, L). 
            species_ids: Species token IDs of shape (B,).
            output_track: Whether to output track predictions.
            inputs_embeds: Pre-computed embeddings of shape (B, L, token_embed_dim).
                           Useful for saliency/attribution analysis.
            output_hidden_states: Whether to return hidden states from each layer.
            output_attentions: Whether to return attention weights from each layer.
            return_dict: Whether to return a ModelOutput instead of a tuple.

        Returns:
            Dictionary containing logits, condition logits, and intermediate embeddings.
        """
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
        return self.core(input_ids = input_ids, 
                         condition_ids = condition_ids, 
                         conditions_masks = conditions_masks, 
                         inputs_embeds = inputs_embeds,
                         output_hidden_states = collect_h,
                         output_attentions = collect_a)
