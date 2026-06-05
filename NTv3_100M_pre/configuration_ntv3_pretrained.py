from __future__ import annotations

import numpy as np
from transformers import PretrainedConfig

class Ntv3PreTrainedConfig(PretrainedConfig):
    """
    Base configuration for NTv3 pre-trained model.

    This architecture uses a convolution tower to downsample the sequence length,
    followed by a Transformer torso and a deconvolution tower to upsample the sequence
    length back to its input size.

    Args:
        alphabet_size: Number of possible tokens.
        pad_token_id: ID of pad token.
        mask_token_id: ID of mask token.
        num_downsamples: Number of times the sequence length is divided by two
            through convolutions before flowing in the Transformer torso.
        attention_heads: Number of heads in the Transformer torso.
        key_size: Key size in the Transformer torso.
        token_embed_dim: Token embedding dimension.
        conv_init_embed_dim: Embedding dimension of first conv layer.
        embed_dim: Embedding dimension in the Transformer torso.
        ffn_embed_dim: Feed forward dimension in the Transformer torso.
        num_layers: Number of Transformer layers.
        layer_norm_eps: Epsilon for layer norm.
        num_hidden_layers_head: Number of hidden layers in head.
        use_skip_connection: Whether to use skip connections in deconv tower.
        use_remat_in_transformer: Whether to use gradient checkpointing in Transformer.
        use_remat_in_convs: Whether to use gradient checkpointing in conv towers.
        deconv_upsample_type: Type of upsampling in deconv tower.
        deconv_phase: Phase for deconv (odd/even).
        embeddings_layers_to_save: Indices of Transformer layers to save embeddings for.
        attention_maps_to_save: Indices of Transformer layers to save attention maps for.
        deconv_layers_to_save: Indices of de-convolution layers to save embeddings for.
    """

    model_type = "ntv3"

    def __init__(
        self,
        # --- Architecture / data ---
        alphabet_size: int = 11,
        pad_token_id: int = 1,
        mask_token_id: int = 2,
        num_downsamples: int = 7,
        attention_heads: int = 8,
        key_size: int = 32,
        token_embed_dim: int = 16,
        conv_init_embed_dim: int = 256,
        embed_dim: int = 256,
        ffn_embed_dim: int = 1024,
        num_layers: int = 2,
        layer_norm_eps: float = 1e-5,
        num_hidden_layers_head: int = 0,
        use_skip_connection: bool = True,
        use_remat_in_transformer: bool = False,
        use_remat_in_convs: bool = False,
        deconv_upsample_type: str = "repeat+conv",  # one of: "conv_transpose", "repeat+conv"
        deconv_phase: str = "odd",
        # --- Return / save intermediate outputs ---
        embeddings_layers_to_save: tuple[int, ...] = (),
        attention_maps_to_save: list = (),  # type: ignore[assignment]
        deconv_layers_to_save: tuple[int, ...] = (),
        # --- Mixed precision knobs (match JAX names, all default to float32) ---
        embedding_compute_dtype: str = "float32",
        embedding_param_dtype: str = "float32",
        stem_compute_dtype: str = "float32",
        stem_param_dtype: str = "float32",
        down_convolution_compute_dtype: str = "float32",
        down_convolution_param_dtype: str = "float32",
        layernorm_compute_dtype: str = "float32",
        layernorm_param_dtype: str = "float32",
        transformer_qkvo_compute_dtype: str = "float32",
        transformer_qkvo_param_dtype: str = "float32",
        transformer_ffn_compute_dtype: str = "float32",
        transformer_ffn_param_dtype: str = "float32",
        lmhead_compute_dtype: str = "float32",
        lmhead_param_dtype: str = "float32",
        up_convolution_compute_dtype: str = "float32",
        up_convolution_param_dtype: str = "float32",
        # present in JAX config; currently a no-op in this PyTorch port but
        # kept for parity and potential future use
        modulation_compute_dtype: str = "float32",
        modulation_param_dtype: str = "float32",
        # HF niceties
        tie_word_embeddings: bool | None = None,
        **kwargs,
    ):
        # Defensive pops to avoid accidental duplication when loading from JSON
        tie_word_embeddings = (
            False if tie_word_embeddings is None else bool(tie_word_embeddings)
        )
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )

        # Store config fields
        self.alphabet_size = int(alphabet_size)
        self.mask_token_id = int(mask_token_id)
        self.num_downsamples = int(num_downsamples)
        self.attention_heads = int(attention_heads)
        self.key_size = int(key_size)
        self.token_embed_dim = int(token_embed_dim)
        self.conv_init_embed_dim = int(conv_init_embed_dim)
        self.embed_dim = int(embed_dim)
        self.ffn_embed_dim = int(ffn_embed_dim)
        self.num_layers = int(num_layers)
        self.layer_norm_eps = float(layer_norm_eps)
        self.num_hidden_layers_head = int(num_hidden_layers_head)
        self.use_skip_connection = bool(use_skip_connection)
        self.use_remat_in_transformer = bool(use_remat_in_transformer)
        self.use_remat_in_convs = bool(use_remat_in_convs)
        self.deconv_upsample_type = str(deconv_upsample_type)
        self.deconv_phase = str(deconv_phase)

        # Return / save intermediate outputs
        self.embeddings_layers_to_save = tuple(embeddings_layers_to_save)
        self.attention_maps_to_save = [tuple(x) for x in attention_maps_to_save]
        self.deconv_layers_to_save = tuple(deconv_layers_to_save)

        # Mixed precision fields (strings: "float32" or "bfloat16")
        self.embedding_compute_dtype = str(embedding_compute_dtype)
        self.embedding_param_dtype = str(embedding_param_dtype)

        self.stem_compute_dtype = str(stem_compute_dtype)
        self.stem_param_dtype = str(stem_param_dtype)

        self.down_convolution_compute_dtype = str(down_convolution_compute_dtype)
        self.down_convolution_param_dtype = str(down_convolution_param_dtype)

        self.layernorm_compute_dtype = str(layernorm_compute_dtype)
        self.layernorm_param_dtype = str(layernorm_param_dtype)

        self.transformer_qkvo_compute_dtype = str(transformer_qkvo_compute_dtype)
        self.transformer_qkvo_param_dtype = str(transformer_qkvo_param_dtype)

        self.transformer_ffn_compute_dtype = str(transformer_ffn_compute_dtype)
        self.transformer_ffn_param_dtype = str(transformer_ffn_param_dtype)

        self.lmhead_compute_dtype = str(lmhead_compute_dtype)
        self.lmhead_param_dtype = str(lmhead_param_dtype)

        self.up_convolution_compute_dtype = str(up_convolution_compute_dtype)
        self.up_convolution_param_dtype = str(up_convolution_param_dtype)

        self.modulation_compute_dtype = str(modulation_compute_dtype)
        self.modulation_param_dtype = str(modulation_param_dtype)

        # Explicitly set for HF compatibility
        self.tie_word_embeddings = bool(tie_word_embeddings)

    # convenience helpers (optional): provide a dict with dtype fields for logging
    @property
    def dtype_summary(self) -> dict[str, str]:
        return {
            k: getattr(self, k)
            for k in [
                "embedding_compute_dtype",
                "stem_compute_dtype",
                "down_convolution_compute_dtype",
                "layernorm_compute_dtype",
                "transformer_qkvo_compute_dtype",
                "transformer_ffn_compute_dtype",
                "up_convolution_compute_dtype",
                "lmhead_compute_dtype",
            ]
        }

    @property
    def filter_list(self) -> list[int]:
        """Compute the filter sizes for conv/deconv towers."""
        return list(
            np.linspace(
                self.conv_init_embed_dim, self.embed_dim, self.num_downsamples + 1
            ).astype(int)
        )


__all__ = [
    "Ntv3PreTrainedConfig",
]

