from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import PreTrainedTokenizer


def _as_str(token: Any) -> Optional[str]:
    """Return the plain string representation for tokens/AddedTokens."""
    if token is None:
        return None
    if isinstance(token, str):
        return token
    return getattr(token, "content", str(token))


class _BaseNTv3Tokenizer(PreTrainedTokenizer):
    """Shared convenience implementation for the NTv3 tokenizers."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids"]

    def __init__(
        self,
        *,
        vocab_file: Optional[str],
        unk_token: str,
        pad_token: str,
        mask_token: str,
        cls_token: str,
        eos_token: str,
        bos_token: str,
        default_tokens: Sequence[str],
        standard_tokens: Iterable[str],
        **kwargs: Any,
    ) -> None:
        if vocab_file is None:
            token_to_id = {tok: idx for idx, tok in enumerate(default_tokens)}
        else:
            if not os.path.isfile(vocab_file):
                raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")
            with open(vocab_file, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                token_to_id = {str(tok): int(idx) for tok, idx in loaded.items()}
            else:
                token_to_id = {str(tok): idx for idx, tok in enumerate(loaded)}

        self._token_to_id: Dict[str, int] = dict(token_to_id)
        self._id_to_token: Dict[int, str] = {
            int(idx): tok for tok, idx in self._token_to_id.items()
        }

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            eos_token=eos_token,
            bos_token=bos_token,
            **kwargs,
        )

        self._unk_token_str = _as_str(self.unk_token)
        self._pad_token_str = _as_str(self.pad_token)
        self._mask_token_str = _as_str(self.mask_token)
        self._cls_token_str = _as_str(self.cls_token)
        self._eos_token_str = _as_str(self.eos_token)
        self._bos_token_str = _as_str(self.bos_token)

        self._special_token_strings = {
            tok
            for tok in (
                self._unk_token_str,
                self._pad_token_str,
                self._mask_token_str,
                self._cls_token_str,
                self._eos_token_str,
                self._bos_token_str,
            )
            if tok is not None
        }

        self._standard_tokens = {str(tok) for tok in standard_tokens}
        self._unk_literal = (
            self._unk_token_str if self._unk_token_str in self._token_to_id else "<unk>"
        )

    # ------------------------------------------------------------------
    # Hugging Face required interface
    # ------------------------------------------------------------------
    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    # Sub-classes implement `_tokenize`.

    def _convert_token_to_id(self, token: str) -> int:
        if self._unk_literal in self._token_to_id:
            return self._token_to_id.get(token, self._token_to_id[self._unk_literal])
        return self._token_to_id.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        idx = int(index)
        return self._id_to_token.get(idx, self._unk_literal)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        length = len(token_ids_0) + (len(token_ids_1) if token_ids_1 else 0)
        if not already_has_special_tokens:
            return [0] * length
        return [
            1 if self._convert_id_to_token(idx) in self._special_token_strings else 0
            for idx in token_ids_0
        ] + (
            [
                1
                if self._convert_id_to_token(idx) in self._special_token_strings
                else 0
                for idx in (token_ids_1 or [])
            ]
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [0] * (len(token_ids_0) + (len(token_ids_1) if token_ids_1 else 0))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        filename = self.vocab_files_names["vocab_file"]
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        path = os.path.join(save_directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._token_to_id, handle, ensure_ascii=False, indent=2)
        return (path,)

    # ------------------------------------------------------------------
    # Optional niceties
    # ------------------------------------------------------------------
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        spaces_between_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        output_tokens: List[str] = []
        for idx in token_ids:
            token = self._convert_id_to_token(idx)
            if skip_special_tokens and token in self._special_token_strings:
                continue
            output_tokens.append(token)
        return self.convert_tokens_to_string(output_tokens)


class NTv3Tokenizer(_BaseNTv3Tokenizer):
    """Character-level tokenizer for NTv3 DNA sequences."""

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        cls_token: str = "<cls>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
        **kwargs: Any,
    ) -> None:
        dna_tokens = ("A", "T", "C", "G", "N")
        default_tokens = (
            unk_token,
            pad_token,
            mask_token,
            cls_token,
            eos_token,
            bos_token,
            *dna_tokens,
        )
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            eos_token=eos_token,
            bos_token=bos_token,
            default_tokens=default_tokens,
            standard_tokens=dna_tokens,
            **kwargs,
        )

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for char in text:
            candidate = char.upper()
            if candidate in self._standard_tokens or candidate in self._token_to_id:
                tokens.append(candidate)
            else:
                tokens.append(self._unk_literal)
        return tokens


__all__ = [
    "NTv3Tokenizer",
]

