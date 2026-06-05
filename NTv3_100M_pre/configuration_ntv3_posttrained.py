from __future__ import annotations

try:
    from .configuration_ntv3_pretrained import Ntv3PreTrainedConfig
except ImportError:
    from configuration_ntv3_pretrained import Ntv3PreTrainedConfig  # type: ignore


class DiscreteConditionedNTv3Config(Ntv3PreTrainedConfig):
    """
    Configuration for NTv3 with discrete token conditioning.

    Adds fields for conditioning the model on discrete tokens (e.g., species IDs).

    Args:
        conditions_vocab_size: List of vocabulary sizes for each condition.
        conditions_names: Optional list of names for each condition.
        conditions_dims: Optional list of embedding dimensions for each condition.
            If not provided, defaults to token_embed_dim for each.
    """

    model_type = "ntv3_conditioned"

    def __init__(
        self,
        conditions_vocab_size: list[int] | None = None,
        conditions_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Conditioning fields ---
        self.conditions_vocab_size = (
            list(conditions_vocab_size) if conditions_vocab_size else []
        )

        # Set default conditions_names if not provided
        if conditions_names is None:
            self.conditions_names = [
                f"condition_{i}" for i in range(len(self.conditions_vocab_size))
            ]
        else:
            self.conditions_names = list(conditions_names)


    @classmethod
    def from_base_config(
        cls,
        config: Ntv3PreTrainedConfig,
        conditions_vocab_size: list[int],
        conditions_names: list[str] | None = None,
    ) -> "DiscreteConditionedNTv3Config":
        """
        Create a DiscreteConditionedNTv3Config from a base NTv3Config.
        """
        parent_dict = config.to_dict()
        parent_dict.pop("model_type", None)
        parent_dict.pop("transformers_version", None)

        return cls(
            conditions_vocab_size=conditions_vocab_size,
            conditions_names=conditions_names,
            **parent_dict,
        )

class NTv3PostTrainedConfig(DiscreteConditionedNTv3Config):
    """
    Configuration for the NTv3 post-trained model with prediction heads.

    Adds fields for multi-species track prediction heads.

    Args:
        bigwigs_per_species: Dictionary mapping species names to lists of
            bigwig track IDs. Used to define the tracks the model predicts.
        bed_elements_names: List of bed element names to predict.
        keep_target_center_fraction: Fraction of sequence center to use for
            track prediction (0.0 to 1.0). Default 1.0 uses full sequence.
        num_species_special_tokens: Offset for species token IDs to head indices.
            E.g., if species tokens start at 6, this should be 6.
    """

    model_type = "ntv3_posttrained"

    def __init__(
        self,
        bigwigs_per_species: dict[str, list[str]] | None = None,
        bed_elements_names: list[str] | None = None,
        species_to_token_id: dict[str, int] | None = None,
        keep_target_center_fraction: float = 1.0,
        num_species_special_tokens: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.bigwigs_per_species = dict(bigwigs_per_species) if bigwigs_per_species else {}
        self.species_to_token_id = dict(species_to_token_id) if species_to_token_id else {}
        self.bed_elements_names = list(bed_elements_names) if bed_elements_names else []
        self.keep_target_center_fraction = float(keep_target_center_fraction)
        self.num_species_special_tokens = int(num_species_special_tokens)

    @classmethod
    def from_conditioned_config(
        cls,
        config: DiscreteConditionedNTv3Config,
        bigwigs_per_species: dict[str, list[str]],
        bed_elements_names: list[str],
        species_to_token_id: dict[str, int],
        keep_target_center_fraction: float = 1.0,
        num_species_special_tokens: int = 6,
    ) -> "NTv3PostTrainedConfig":
        """
        Create an NTv3PostTrainedConfig from a DiscreteConditionedNTv3Config.
        """
        parent_dict = config.to_dict()
        parent_dict.pop("model_type", None)
        parent_dict.pop("transformers_version", None)

        return cls(
            bigwigs_per_species=bigwigs_per_species,
            bed_elements_names=bed_elements_names,
            species_to_token_id=species_to_token_id,
            keep_target_center_fraction=keep_target_center_fraction,
            num_species_special_tokens=num_species_special_tokens,
            **parent_dict,
        )


__all__ = [
    "Ntv3PreTrainedConfig",
    "DiscreteConditionedNTv3Config",
    "NTv3PostTrainedConfig",
]

