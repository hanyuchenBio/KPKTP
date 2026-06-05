#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate embeddings and predict Klebsiella K types.

Main outputs:
    1. Positive K-type ranking table specified by -o/--output.
    2. Optional full prediction table specified by -out/--out.

The ranking table is built from the full prediction table: for each sequence,
only K types predicted as positive are kept, then they are sorted by ascending
error probability.

Embedding-to-model feature matching is based only on numeric column order.
Saved feature column names in model metadata are ignored.

K-type and Platt model bundles must be provided by -km and -pm.

Transformer-related settings are fixed inside this script:
    Protein encoder = ESM-C 300M local .pth weight
    Protein embedding dim = 960
    Nucleotide encoder = NTv3 100M local model
    NTv3 window size = 8192
    NTv3 stride = 4096
    Nucleotide Transformer batch size = 8
    Device = automatic cuda/cpu detection
"""

from __future__ import annotations

import os
import argparse
import csv
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FASTA_EXTENSIONS = {".fa", ".fasta", ".faa", ".fna", ".fas", ".fsa"}

DEFAULT_ESMC_WEIGHT_NAME = "esmc_300m_2024_12_v0.pth"
DEFAULT_ESMC_HF_HOME = "/data/hanyuchen2/hf_cache"
DEFAULT_ESMC_REPO_CACHE_NAME = "models--EvolutionaryScale--esmc-300m-2024-12"
DEFAULT_NTV3_MODEL_DIR_NAME = "NTv3_100M_pre"

FIXED_PROTEIN_BATCH_SIZE = 1
FIXED_NUCLEOTIDE_BATCH_SIZE = 8
FIXED_NTV3_WINDOW_SIZE = 8192
FIXED_NTV3_STRIDE = 4096
FIXED_ESMC_USE_FLASH_ATTN = False


np = None
pd = None
torch = None
joblib = None
esm = None
AutoTokenizer = None
AutoModelForMaskedLM = None

SKLEARN_VERSION_WARNING_PRINTED = False


@dataclass
class FastaRecord:
    seq_id: str
    seq: str
    source_file: str


@dataclass
class FilePair:
    pair_id: str
    protein_file: Path
    nucleotide_file: Path


@dataclass
class ProteinRecord:
    seq_id: str
    seq: str
    source_file: str
    starts: Optional[List[int]] = None
    coverage: Optional[Any] = None
    embedding: Optional[Any] = None


@dataclass
class LoadedModel:
    model_key: str
    model: object
    k_name: str
    embedding_kind: str
    embedding_name: str
    feature_cols: List[str]
    threshold: float = 0.5


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("-p", "--protein", default=None, help="Protein FASTA file or folder. ESM-C 300M is used.")
    parser.add_argument("-n", "--nucleotide", default=None, help="Nucleotide FASTA file or folder. NTv3 100M is used.")
    parser.add_argument("-csv", "--csv", default=None, help="Optional pairing CSV. Column 1 = protein file name; column 2 = nucleotide file name.")
    parser.add_argument("-km", "--ktype-model", required=True, help="Required trained K-type model bundle file, usually final_rf_1to1_model_bundle.joblib.")
    parser.add_argument("-pm", "--platt-model", required=True, help="Required trained Platt calibration model bundle file, usually platt_bundle_models.joblib.")
    parser.add_argument("-o", "--output", dest="rank_output", required=True, help="Required positive K-type ranking table path, usually .csv.")
    parser.add_argument("-out", "--out", dest="full_output", default=None, help="Optional full prediction table path, usually .csv.")
    return parser


def print_simple_usage() -> None:
    script = Path(sys.argv[0]).name
    print("Usage:")
    print(f"  python {script} -p <protein_fasta_or_folder> -km <k_model.joblib> -pm <platt_model.joblib> -o <rank.csv>")
    print(f"  python {script} -n <nucleotide_fasta_or_folder> -km <k_model.joblib> -pm <platt_model.joblib> -o <rank.csv>")
    print(f"  python {script} -p <protein_fasta_or_folder> -n <nucleotide_fasta_or_folder> -km <k_model.joblib> -pm <platt_model.joblib> -o <rank.csv>")
    print(f"  python {script} -p <protein_fasta_or_folder> -n <nucleotide_fasta_or_folder> -csv <pair_table.csv> -km <k_model.joblib> -pm <platt_model.joblib> -o <rank.csv> -out <full_prediction.csv>")
    print()
    print("Parameters:")
    print("  -p,    --protein             Protein FASTA file or folder.")
    print("  -n,    --nucleotide          Nucleotide FASTA file or folder.")
    print("  -csv,  --csv                 Optional pairing CSV: column 1 protein file name, column 2 nucleotide file name.")
    print("  -km,   --ktype-model         Required trained K-type model bundle file.")
    print("  -pm,   --platt-model         Required trained Platt calibration model bundle file.")
    print("  -o,    --output              Required positive K-type ranking table.")
    print("  -out,  --out                 Optional full prediction table.")
    print()
    print("Fixed internal settings:")
    print("  Protein encoder: ESM-C 300M local .pth")
    print("  Protein embedding dim: 960")
    print(f"  ESM-C sequence batch size: {FIXED_PROTEIN_BATCH_SIZE}")
    print("  ESM-C Flash Attention: disabled by default")
    print("  Nucleotide encoder: NTv3 100M local model")
    print(f"  NTv3 window size: {FIXED_NTV3_WINDOW_SIZE}")
    print(f"  NTv3 stride: {FIXED_NTV3_STRIDE}")
    print(f"  NTv3 window batch size: {FIXED_NUCLEOTIDE_BATCH_SIZE}")
    print("  Device: automatic cuda/cpu detection")


def parse_args() -> argparse.Namespace:
    parser = build_argparser()

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        missing_items = collect_help_missing_items()
        if missing_items:
            for item in missing_items:
                print(item)
        else:
            print_simple_usage()
        sys.exit(0)

    return parser.parse_args()


def validate_basic_args(args: argparse.Namespace) -> None:
    if args.protein is None and args.nucleotide is None:
        raise ValueError("At least one input must be provided: -p protein FASTA and/or -n nucleotide FASTA.")
    if args.csv is not None and not (args.protein is not None and args.nucleotide is not None):
        raise ValueError("-csv can be used only when both -p and -n are provided.")

    ktype_model_path = Path(args.ktype_model)
    if not ktype_model_path.is_file():
        raise FileNotFoundError(f"-km/--ktype-model must be an existing model file: {ktype_model_path}")

    platt_model_path = Path(args.platt_model)
    if not platt_model_path.is_file():
        raise FileNotFoundError(f"-pm/--platt-model must be an existing Platt model file: {platt_model_path}")

    rank_output_path = Path(args.rank_output)
    if rank_output_path.exists() and rank_output_path.is_dir():
        raise IsADirectoryError(f"-o/--output must be a file path, not a directory: {rank_output_path}")

    if args.full_output is not None:
        full_output_path = Path(args.full_output)
        if full_output_path.exists() and full_output_path.is_dir():
            raise IsADirectoryError(f"-out/--out must be a file path, not a directory: {full_output_path}")


def find_named_files_current_tree(file_name: str, preferred_relative_paths: Optional[List[str]] = None) -> List[Path]:
    """Find files by name under the current working directory only."""
    root = Path.cwd().resolve()
    matches: List[Path] = []

    for rel in preferred_relative_paths or []:
        candidate = (root / rel).resolve()
        if candidate.is_file() and candidate.name == file_name:
            matches.append(candidate)

    for candidate in sorted(root.rglob(file_name)):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in matches:
            matches.append(resolved)

    return matches


def find_named_dirs_current_tree(dir_name: str, preferred_relative_paths: Optional[List[str]] = None) -> List[Path]:
    """Find directories by name under the current working directory only."""
    root = Path.cwd().resolve()
    matches: List[Path] = []

    for rel in preferred_relative_paths or []:
        candidate = (root / rel).resolve()
        if candidate.is_dir() and candidate.name == dir_name:
            matches.append(candidate)

    for candidate in sorted(root.rglob(dir_name)):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved.name == dir_name and resolved not in matches:
            matches.append(resolved)

    return matches


def discover_ntv3_model_dir() -> Path:
    matches = find_named_dirs_current_tree(
        DEFAULT_NTV3_MODEL_DIR_NAME,
        preferred_relative_paths=[
            f"models/{DEFAULT_NTV3_MODEL_DIR_NAME}",
            DEFAULT_NTV3_MODEL_DIR_NAME,
        ],
    )
    if not matches:
        raise FileNotFoundError(
            "Required NTv3 100M local model directory was not found under the current working directory.\n"
            f"Required directory name: {DEFAULT_NTV3_MODEL_DIR_NAME}\n"
            f"Current working directory searched: {Path.cwd().resolve()}"
        )
    if len(matches) > 1:
        print(f"[WARN] Multiple {DEFAULT_NTV3_MODEL_DIR_NAME} directories found. Using the first one: {matches[0]}")
    else:
        print(f"[INFO] NTv3 100M model directory found: {matches[0]}")
    return matches[0]


def discover_esmc300m_weight_file() -> Path:
    matches = find_named_files_current_tree(
        DEFAULT_ESMC_WEIGHT_NAME,
        preferred_relative_paths=[
            f"models/{DEFAULT_ESMC_WEIGHT_NAME}",
            f"esmc/{DEFAULT_ESMC_WEIGHT_NAME}",
            f"esm/{DEFAULT_ESMC_WEIGHT_NAME}",
            f"weights/{DEFAULT_ESMC_WEIGHT_NAME}",
            DEFAULT_ESMC_WEIGHT_NAME,
        ],
    )

    hf_home = Path(DEFAULT_ESMC_HF_HOME).expanduser().resolve()
    repo_cache = hf_home / "hub" / DEFAULT_ESMC_REPO_CACHE_NAME
    hf_candidates: List[Path] = []

    snapshots_dir = repo_cache / "snapshots"
    if snapshots_dir.is_dir():
        hf_candidates.extend(snapshots_dir.glob(f"*/data/weights/{DEFAULT_ESMC_WEIGHT_NAME}"))
    if repo_cache.is_dir():
        hf_candidates.extend(repo_cache.glob(f"**/{DEFAULT_ESMC_WEIGHT_NAME}"))

    seen = {p.resolve() for p in matches if p.exists()}
    for candidate in hf_candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in seen and candidate.is_file():
            matches.append(candidate)
            seen.add(resolved)

    if not matches:
        raise FileNotFoundError(
            "Required ESM-C 300M local weight file was not found.\n"
            f"Required file name: {DEFAULT_ESMC_WEIGHT_NAME}\n"
            f"Current working directory searched: {Path.cwd().resolve()}\n"
            f"Hugging Face cache searched: {hf_home}"
        )

    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    if len(matches) > 1:
        print(f"[WARN] Multiple ESM-C 300M weight files found. Using the newest one: {matches[0]}")
    else:
        print(f"[INFO] ESM-C 300M weight file found: {matches[0]}")
    return matches[0]


def collect_help_missing_items() -> List[str]:
    """Return only missing package names or required file/directory names for -h."""
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("joblib", "joblib"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
        ("esm", "esm"),
        ("transformers", "transformers"),
        ("safetensors", "safetensors"),
    ]

    missing: List[str] = []

    for module_name, package_name in required_packages:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    esmc_matches = find_named_files_current_tree(
        DEFAULT_ESMC_WEIGHT_NAME,
        preferred_relative_paths=[
            f"models/{DEFAULT_ESMC_WEIGHT_NAME}",
            f"esmc/{DEFAULT_ESMC_WEIGHT_NAME}",
            f"esm/{DEFAULT_ESMC_WEIGHT_NAME}",
            f"weights/{DEFAULT_ESMC_WEIGHT_NAME}",
            DEFAULT_ESMC_WEIGHT_NAME,
        ],
    )
    hf_home = Path(DEFAULT_ESMC_HF_HOME).expanduser().resolve()
    repo_cache = hf_home / "hub" / DEFAULT_ESMC_REPO_CACHE_NAME
    hf_match_exists = False
    snapshots_dir = repo_cache / "snapshots"
    if snapshots_dir.is_dir():
        hf_match_exists = any(snapshots_dir.glob(f"*/data/weights/{DEFAULT_ESMC_WEIGHT_NAME}"))
    if not hf_match_exists and repo_cache.is_dir():
        hf_match_exists = any(repo_cache.glob(f"**/{DEFAULT_ESMC_WEIGHT_NAME}"))
    if not esmc_matches and not hf_match_exists:
        missing.append(DEFAULT_ESMC_WEIGHT_NAME)

    nt_matches = find_named_dirs_current_tree(
        DEFAULT_NTV3_MODEL_DIR_NAME,
        preferred_relative_paths=[f"models/{DEFAULT_NTV3_MODEL_DIR_NAME}", DEFAULT_NTV3_MODEL_DIR_NAME],
    )
    if not nt_matches:
        missing.append(DEFAULT_NTV3_MODEL_DIR_NAME)

    return missing


def import_runtime_modules() -> None:
    global np, pd, torch, joblib, esm, AutoTokenizer, AutoModelForMaskedLM

    np = importlib.import_module("numpy")
    pd = importlib.import_module("pandas")
    torch = importlib.import_module("torch")
    joblib = importlib.import_module("joblib")

    if importlib.util.find_spec("esm") is not None:
        esm = importlib.import_module("esm")
    if importlib.util.find_spec("transformers") is not None:
        transformers = importlib.import_module("transformers")
        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForMaskedLM = transformers.AutoModelForMaskedLM


def choose_device(device_arg: Optional[str]) -> Any:
    if device_arg is not None:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available in this environment.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def joblib_load_with_compact_sklearn_warning(path: Path, label: str) -> Any:
    """Load a joblib object and reduce repeated scikit-learn version warnings.

    The original warning is not a runtime failure, but scikit-learn prints it once
    for every nested estimator in a RandomForest or Platt bundle. This function
    keeps the run log readable by replacing those repeated messages with one
    concise warning. Real loading errors are still raised normally.
    """
    global SKLEARN_VERSION_WARNING_PRINTED

    sklearn_warning_cls = None
    try:
        if importlib.util.find_spec("sklearn") is not None:
            sklearn_exceptions = importlib.import_module("sklearn.exceptions")
            sklearn_warning_cls = getattr(sklearn_exceptions, "InconsistentVersionWarning", None)
    except Exception:
        sklearn_warning_cls = None

    caught_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = joblib.load(path)
        caught_warnings = list(caught)

    for warning_item in caught_warnings:
        if sklearn_warning_cls is not None and issubclass(warning_item.category, sklearn_warning_cls):
            if not SKLEARN_VERSION_WARNING_PRINTED:
                print(
                    "[WARN] scikit-learn version mismatch detected while loading joblib models; "
                    "repeated InconsistentVersionWarning messages were suppressed. "
                    "For strict reproducibility, use the same scikit-learn version used during training."
                )
                SKLEARN_VERSION_WARNING_PRINTED = True
        else:
            warnings.warn(warning_item.message, warning_item.category)

    return obj


def read_fasta(fasta_file: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    seq_id: Optional[str] = None
    seq_parts: List[str] = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_parts)))
                seq_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())

    if seq_id is not None:
        records.append((seq_id, "".join(seq_parts)))
    return records


def find_fasta_files(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in FASTA_EXTENSIONS:
            raise ValueError(f"Unsupported FASTA extension: {input_path}")
        return [input_path]
    files = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in FASTA_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No FASTA files found in: {input_path}")
    return files


def make_unique_id(seq_id: str, seen_ids: Dict[str, int]) -> str:
    if seq_id not in seen_ids:
        seen_ids[seq_id] = 1
        return seq_id
    seen_ids[seq_id] += 1
    return f"{seq_id}__dup{seen_ids[seq_id]}"


def load_fasta_records(input_path: str, sequence_type: str) -> List[FastaRecord]:
    path = Path(input_path)
    input_is_dir = path.is_dir()
    fasta_files = find_fasta_files(path)
    records: List[FastaRecord] = []
    seen_ids: Dict[str, int] = {}
    skipped_empty = 0

    for fasta_file in fasta_files:
        file_records = read_fasta(fasta_file)
        if not file_records:
            print(f"[WARN] No FASTA records found in: {fasta_file}")
            continue

        for raw_id, raw_seq in file_records:
            seq = raw_seq.strip().upper()
            if not seq:
                skipped_empty += 1
                print(f"[WARN] Empty sequence skipped: {fasta_file} | {raw_id}")
                continue

            if input_is_dir:
                output_id = fasta_file.stem if len(file_records) == 1 else f"{fasta_file.stem}|{raw_id}"
            else:
                output_id = raw_id

            output_id = make_unique_id(output_id, seen_ids)
            records.append(FastaRecord(seq_id=output_id, seq=seq, source_file=str(fasta_file)))

    if not records:
        raise ValueError(f"No valid {sequence_type} sequences were loaded from: {input_path}")

    print(f"[INFO] {sequence_type} FASTA files found: {len(fasta_files)}")
    print(f"[INFO] {sequence_type} sequences loaded: {len(records)}")
    print(f"[INFO] {sequence_type} empty sequences skipped: {skipped_empty}")
    return records



def strip_fasta_extension(name: str) -> str:
    return re.sub(r"\.(fa|fasta|faa|fna|fas|fsa)$", "", name, flags=re.IGNORECASE)


def strip_common_sequence_suffixes(name: str) -> str:
    """Remove common protein/nucleotide suffix tokens for file matching only."""
    text = str(name).strip().lower()
    text = re.sub(r"\s+", "_", text)

    suffix_patterns = [
        r"([._-])(cds|cds_sequence|coding_sequence|nt|nuc|nucleotide|nucleotides|dna|rna|fna)$",
        r"([._-])(protein|prot|aa|pep|peptide|faa)$",
    ]

    changed = True
    while changed:
        old = text
        for pattern in suffix_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        changed = text != old

    return text


def file_key_candidates(value: object) -> List[str]:
    """Generate tolerant file-name keys for protein/nucleotide pairing.

    Matching order is conservative first, then tolerant:
    1. full file name with extension
    2. file name without FASTA extension
    3. common suffix-normalized name, e.g. A_cds -> A, A_protein -> A
    4. weak final-token removal, e.g. A_p/A_n -> A
    """
    raw = str(value).strip().replace("\\", "/")
    base = Path(raw).name.strip().lower()
    if not base:
        return []

    no_ext = strip_fasta_extension(base).lower()
    normalized = strip_common_sequence_suffixes(no_ext)
    weak = re.sub(r"([._-])(p|n)$", "", normalized, flags=re.IGNORECASE)

    candidates: List[str] = []
    for key in [base, no_ext, normalized, weak]:
        key = str(key).strip().lower()
        if key and key not in candidates:
            candidates.append(key)
    return candidates


def normalize_file_key(value: object) -> str:
    candidates = file_key_candidates(value)
    return candidates[-1] if candidates else ""


def build_fasta_lookup(input_path: str, label: str) -> Tuple[Dict[str, Optional[Path]], List[Path]]:
    """Build a tolerant lookup for FASTA files.

    Ambiguous keys are kept as None rather than raising an error. This allows
    the pipeline to continue with uniquely matched files and report the rest as
    unmatched.
    """
    files = find_fasta_files(Path(input_path))
    lookup: Dict[str, Optional[Path]] = {}
    ambiguous: Dict[str, List[Path]] = {}

    for file_path in files:
        for key in file_key_candidates(file_path.name):
            if key not in lookup:
                lookup[key] = file_path
            elif lookup[key] is not None and lookup[key] != file_path:
                ambiguous[key] = [lookup[key], file_path]
                lookup[key] = None
            elif lookup[key] is None:
                ambiguous.setdefault(key, []).append(file_path)

    if ambiguous:
        print(f"[WARN] Ambiguous {label} file-name keys were detected. Ambiguous keys will not be used for pairing.")
        for index, (key, paths) in enumerate(sorted(ambiguous.items()), start=1):
            if index > 20:
                print("  ...")
                break
            unique_paths = []
            for path in paths:
                if path not in unique_paths:
                    unique_paths.append(path)
            print(f"  key={key} -> " + "; ".join(p.name for p in unique_paths))

    return lookup, files


def resolve_fasta_from_lookup(file_name: str, lookup: Dict[str, Optional[Path]]) -> Optional[Path]:
    for key in file_key_candidates(file_name):
        path = lookup.get(key)
        if path is not None:
            return path
    return None


def build_unique_key_map(files: List[Path]) -> Dict[str, Optional[Path]]:
    key_map: Dict[str, Optional[Path]] = {}
    for file_path in files:
        for key in file_key_candidates(file_path.name):
            if key not in key_map:
                key_map[key] = file_path
            elif key_map[key] is not None and key_map[key] != file_path:
                key_map[key] = None
    return key_map


def choose_matching_file(
    query_file: Path,
    target_key_map: Dict[str, Optional[Path]],
    used_targets: set,
) -> Tuple[Optional[Path], str]:
    ambiguous_seen = False
    for key in file_key_candidates(query_file.name):
        if key not in target_key_map:
            continue
        target = target_key_map[key]
        if target is None:
            ambiguous_seen = True
            continue
        if target in used_targets:
            return None, "matching_file_already_used"
        return target, ""
    if ambiguous_seen:
        return None, "matching_file_ambiguous_after_normalization"
    return None, "matching_file_not_found"


def read_pair_csv(pair_csv: str) -> List[Tuple[str, str]]:
    csv_path = Path(pair_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Pairing CSV was not found: {csv_path}")

    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if sample.strip():
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
        else:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for line_no, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue
            protein_name = row[0].strip()
            nucleotide_name = row[1].strip()
            if not protein_name or not nucleotide_name:
                continue
            lower_pair = (protein_name.lower(), nucleotide_name.lower())
            if line_no == 1 and (
                lower_pair[0] in {"protein", "protein_file", "protein.fasta", "p", "p_file"}
                or lower_pair[1] in {"nucleotide", "nucleotide_file", "cds", "cds.fasta", "n", "n_file"}
            ):
                continue
            rows.append((protein_name, nucleotide_name))

    if not rows:
        raise ValueError(f"No usable protein/nucleotide file pairs were found in: {csv_path}")
    return rows


def make_unique_pair_id(base_id: str, seen: Dict[str, int]) -> str:
    base_id = str(base_id).strip() or "pair"
    if base_id not in seen:
        seen[base_id] = 1
        return base_id
    seen[base_id] += 1
    return f"{base_id}__dup{seen[base_id]}"



def build_file_pairs(
    protein_input: str,
    nucleotide_input: str,
    pair_csv: Optional[str],
) -> Tuple[List[FilePair], List[Dict[str, str]]]:
    """Build matched protein/nucleotide file pairs.

    The matching is partial-tolerant. The script processes only matched pairs
    and reports all unmatched records. File-name matching accepts common cases
    such as:
        A.faa      <-> A.fna
        A.fasta    <-> A_cds.fasta
        A_protein.fa <-> A_nucleotide.fa
        A_p.fasta  <-> A_n.fasta
    """
    protein_lookup, protein_files = build_fasta_lookup(protein_input, "protein")
    nucleotide_lookup, nucleotide_files = build_fasta_lookup(nucleotide_input, "nucleotide")
    pairs: List[FilePair] = []
    unmatched: List[Dict[str, str]] = []
    seen_pair_ids: Dict[str, int] = {}
    used_protein_files = set()
    used_nucleotide_files = set()

    if pair_csv is not None:
        for row_index, (protein_name, nucleotide_name) in enumerate(read_pair_csv(pair_csv), start=1):
            protein_file = resolve_fasta_from_lookup(protein_name, protein_lookup)
            nucleotide_file = resolve_fasta_from_lookup(nucleotide_name, nucleotide_lookup)

            reasons: List[str] = []
            if protein_file is None:
                reasons.append("protein_file_not_found_or_ambiguous")
            if nucleotide_file is None:
                reasons.append("nucleotide_file_not_found_or_ambiguous")
            if protein_file is not None and protein_file in used_protein_files:
                reasons.append("protein_file_repeated_in_pairs")
            if nucleotide_file is not None and nucleotide_file in used_nucleotide_files:
                reasons.append("nucleotide_file_repeated_in_pairs")

            if reasons:
                unmatched.append(
                    {
                        "source": "csv",
                        "row": str(row_index),
                        "protein_file": protein_name,
                        "nucleotide_file": nucleotide_name,
                        "reason": ";".join(reasons),
                    }
                )
                continue

            pair_id_base = normalize_file_key(protein_name) or Path(protein_name).stem
            pair_id = make_unique_pair_id(pair_id_base, seen_pair_ids)
            pairs.append(FilePair(pair_id=pair_id, protein_file=protein_file, nucleotide_file=nucleotide_file))
            used_protein_files.add(protein_file)
            used_nucleotide_files.add(nucleotide_file)

        # Also report files present in folders but not used by the CSV mapping.
        for protein_file in protein_files:
            if protein_file not in used_protein_files:
                unmatched.append(
                    {
                        "source": "csv_unused_folder_file",
                        "row": "",
                        "protein_file": protein_file.name,
                        "nucleotide_file": "",
                        "reason": "protein_file_not_used_by_matched_csv_pairs",
                    }
                )
        for nucleotide_file in nucleotide_files:
            if nucleotide_file not in used_nucleotide_files:
                unmatched.append(
                    {
                        "source": "csv_unused_folder_file",
                        "row": "",
                        "protein_file": "",
                        "nucleotide_file": nucleotide_file.name,
                        "reason": "nucleotide_file_not_used_by_matched_csv_pairs",
                    }
                )
    else:
        nucleotide_key_map = build_unique_key_map(nucleotide_files)

        for protein_file in sorted(protein_files, key=lambda p: p.name):
            nucleotide_file, reason = choose_matching_file(
                query_file=protein_file,
                target_key_map=nucleotide_key_map,
                used_targets=used_nucleotide_files,
            )

            if nucleotide_file is None:
                unmatched.append(
                    {
                        "source": "auto_filename_normalized",
                        "row": "",
                        "protein_file": protein_file.name,
                        "nucleotide_file": "",
                        "reason": reason or "matching_nucleotide_file_not_found",
                    }
                )
                continue

            pair_id_base = normalize_file_key(protein_file.name) or protein_file.stem
            pair_id = make_unique_pair_id(pair_id_base, seen_pair_ids)
            pairs.append(FilePair(pair_id=pair_id, protein_file=protein_file, nucleotide_file=nucleotide_file))
            used_protein_files.add(protein_file)
            used_nucleotide_files.add(nucleotide_file)

        for nucleotide_file in sorted(nucleotide_files, key=lambda p: p.name):
            if nucleotide_file not in used_nucleotide_files:
                unmatched.append(
                    {
                        "source": "auto_filename_normalized",
                        "row": "",
                        "protein_file": "",
                        "nucleotide_file": nucleotide_file.name,
                        "reason": "matching_protein_file_not_found_or_already_used",
                    }
                )

    return pairs, unmatched

def print_ready_file_pairs(file_pairs: List[FilePair]) -> None:
    print("[INFO] Ready protein/nucleotide file pairs to process:")
    for index, pair in enumerate(file_pairs, start=1):
        print(f"  {index}. protein={pair.protein_file.name}    nucleotide={pair.nucleotide_file.name}")


def print_unmatched_file_pairs(unmatched_pairs: List[Dict[str, str]]) -> None:
    if not unmatched_pairs:
        print("[INFO] Unmatched protein/nucleotide files: 0")
        return

    print(f"[WARN] Unmatched protein/nucleotide files: {len(unmatched_pairs)}")
    for index, item in enumerate(unmatched_pairs, start=1):
        protein_file = item.get("protein_file", "") or "-"
        nucleotide_file = item.get("nucleotide_file", "") or "-"
        reason = item.get("reason", "") or "unmatched"
        row = item.get("row", "")
        row_text = f" row={row}" if row else ""
        print(f"  {index}.{row_text} protein={protein_file}    nucleotide={nucleotide_file}    reason={reason}")


def write_unmatched_file_pairs(unmatched_pairs: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "row", "protein_file", "nucleotide_file", "reason"])
        for item in unmatched_pairs:
            writer.writerow([
                item.get("source", ""),
                item.get("row", ""),
                item.get("protein_file", ""),
                item.get("nucleotide_file", ""),
                item.get("reason", ""),
            ])


def write_selected_pairs(file_pairs: List[FilePair], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "protein_file", "nucleotide_file"])
        for pair in file_pairs:
            writer.writerow([pair.pair_id, str(pair.protein_file), str(pair.nucleotide_file)])


def normalize_record_key(value: str) -> str:
    text = str(value).strip()
    text = Path(text).name
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\.(fa|fasta|faa|fna|fas|fsa)$", "", text, flags=re.IGNORECASE)
    text = normalize_file_key(text)
    return text.lower()


def pair_records_within_files(
    pair: FilePair,
    p_records: List[Tuple[str, str]],
    n_records: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Return paired record triples: (seq_id_suffix, protein_seq, nucleotide_seq)."""
    warnings_list: List[str] = []
    paired: List[Tuple[str, str, str]] = []

    if len(p_records) == len(n_records):
        for idx, ((p_id, p_seq), (n_id, n_seq)) in enumerate(zip(p_records, n_records), start=1):
            suffix = "" if len(p_records) == 1 else f"|record_{idx}"
            paired.append((suffix, p_seq, n_seq))
        return paired, warnings_list

    p_by_key: Dict[str, Tuple[str, str]] = {}
    n_by_key: Dict[str, Tuple[str, str]] = {}
    duplicated_p: set = set()
    duplicated_n: set = set()

    for p_id, p_seq in p_records:
        key = normalize_record_key(p_id)
        if key in p_by_key:
            duplicated_p.add(key)
        else:
            p_by_key[key] = (p_id, p_seq)

    for n_id, n_seq in n_records:
        key = normalize_record_key(n_id)
        if key in n_by_key:
            duplicated_n.add(key)
        else:
            n_by_key[key] = (n_id, n_seq)

    shared_keys = sorted((set(p_by_key) & set(n_by_key)) - duplicated_p - duplicated_n)
    if shared_keys:
        for idx, key in enumerate(shared_keys, start=1):
            p_id, p_seq = p_by_key[key]
            n_id, n_seq = n_by_key[key]
            suffix = "" if len(shared_keys) == 1 else f"|record_{idx}"
            paired.append((suffix, p_seq, n_seq))
        skipped_p = len(p_records) - len(shared_keys)
        skipped_n = len(n_records) - len(shared_keys)
        if skipped_p or skipped_n:
            warnings_list.append(
                f"{pair.protein_file.name} / {pair.nucleotide_file.name}: "
                f"record IDs matched {len(shared_keys)} pair(s); skipped protein={skipped_p}, nucleotide={skipped_n}."
            )
        return paired, warnings_list

    if len(p_records) == 1 and len(n_records) > 1:
        p_id, p_seq = p_records[0]
        n_id, n_seq = n_records[0]
        paired.append(("", p_seq, n_seq))
        warnings_list.append(
            f"{pair.protein_file.name} has 1 record but {pair.nucleotide_file.name} has {len(n_records)} records; "
            f"record IDs did not match, so the first nucleotide record was used and the remaining {len(n_records) - 1} nucleotide record(s) were skipped."
        )
        return paired, warnings_list

    if len(n_records) == 1 and len(p_records) > 1:
        p_id, p_seq = p_records[0]
        n_id, n_seq = n_records[0]
        paired.append(("", p_seq, n_seq))
        warnings_list.append(
            f"{pair.nucleotide_file.name} has 1 record but {pair.protein_file.name} has {len(p_records)} records; "
            f"record IDs did not match, so the first protein record was used and the remaining {len(p_records) - 1} protein record(s) were skipped."
        )
        return paired, warnings_list

    warnings_list.append(
        f"{pair.protein_file.name} / {pair.nucleotide_file.name}: no record-level match found "
        f"because protein records={len(p_records)}, nucleotide records={len(n_records)}. This file pair was skipped."
    )
    return paired, warnings_list


def load_paired_fasta_records(file_pairs: List[FilePair]) -> Tuple[List[FastaRecord], List[FastaRecord]]:
    protein_records: List[FastaRecord] = []
    nucleotide_records: List[FastaRecord] = []
    record_warnings: List[str] = []

    for pair in file_pairs:
        p_records = read_fasta(pair.protein_file)
        n_records = read_fasta(pair.nucleotide_file)
        if not p_records:
            record_warnings.append(f"No protein FASTA records found in: {pair.protein_file}")
            continue
        if not n_records:
            record_warnings.append(f"No nucleotide FASTA records found in: {pair.nucleotide_file}")
            continue

        paired_records, warnings_for_pair = pair_records_within_files(pair, p_records, n_records)
        record_warnings.extend(warnings_for_pair)

        for idx, (suffix, p_seq, n_seq) in enumerate(paired_records, start=1):
            seq_id = f"{pair.pair_id}{suffix}"
            protein_records.append(FastaRecord(seq_id=seq_id, seq=p_seq.strip().upper(), source_file=str(pair.protein_file)))
            nucleotide_records.append(FastaRecord(seq_id=seq_id, seq=n_seq.strip().upper(), source_file=str(pair.nucleotide_file)))

    for message in record_warnings:
        print(f"[WARN] {message}")

    if not protein_records or not nucleotide_records:
        raise ValueError("No valid paired protein/nucleotide records were loaded after record-level matching.")

    print(f"[INFO] Paired records loaded: {len(protein_records)}")
    return protein_records, nucleotide_records


def clean_protein_sequence(seq: str) -> str:
    valid_aas = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
    return "".join(aa if aa in valid_aas else "X" for aa in seq.upper())


def clean_nucleotide_sequence(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    valid = {"A", "T", "C", "G", "N"}
    return "".join(base if base in valid else "N" for base in seq)


def torch_load_trusted_checkpoint(path: Path):
    """Load a trusted local PyTorch checkpoint across PyTorch versions."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_local_esmc300m(weight_path: Path, device: Any, use_flash_attn: bool = FIXED_ESMC_USE_FLASH_ATTN):
    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
    except Exception as exc:
        raise ImportError(
            "The ESM-C package is required for protein embedding. The installed 'esm' package must provide "
            "esm.models.esmc. Install the EvolutionaryScale ESM package used by ESM-C."
        ) from exc

    print("[INFO] Building ESM-C 300M architecture locally")
    try:
        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    except Exception as exc:
        if use_flash_attn:
            print(f"[WARN] Failed to build ESM-C with Flash Attention ({exc}). Retrying without Flash Attention.")
            model = ESMC(
                d_model=960,
                n_heads=15,
                n_layers=30,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=False,
            ).eval()
        else:
            raise

    print(f"[INFO] Loading local ESM-C 300M weight file: {weight_path}")
    state_dict = torch_load_trusted_checkpoint(weight_path)
    model.load_state_dict(state_dict)

    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    model.eval()
    return model


def mean_pool_esmc_embedding(model: Any, sequence: str) -> Any:
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)

    with torch.no_grad():
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )

    embeddings = output.embeddings
    if embeddings is None:
        raise RuntimeError("No embeddings were returned by ESM-C.")

    embeddings = embeddings.detach().float().cpu()
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(0)
    if embeddings.ndim != 2:
        raise RuntimeError(f"Unexpected ESM-C embedding shape: {tuple(embeddings.shape)}")

    seq_len = len(sequence)
    if embeddings.shape[0] == seq_len + 2:
        embeddings = embeddings[1:-1]
    elif embeddings.shape[0] == seq_len + 1:
        embeddings = embeddings[1:]
    elif embeddings.shape[0] > seq_len:
        embeddings = embeddings[:seq_len]

    if embeddings.shape[0] == 0:
        raise RuntimeError("Empty residue embedding after removing ESM-C special tokens.")

    return embeddings.mean(dim=0)


def embed_protein_records_esmc(
    raw_records: List[FastaRecord],
    device: Any,
    batch_size: int,
    esmc_weight_file: Path,
) -> Any:
    records: List[FastaRecord] = []
    skipped_empty = 0
    for record in raw_records:
        cleaned = clean_protein_sequence(record.seq)
        if cleaned:
            records.append(FastaRecord(seq_id=record.seq_id, seq=cleaned, source_file=record.source_file))
        else:
            skipped_empty += 1

    if not records:
        raise ValueError("No valid protein sequences remained after cleaning.")

    print(f"[INFO] Using device for ESM-C 300M: {device}")
    print(f"[INFO] Protein sequences to embed: {len(records)}")
    if skipped_empty:
        print(f"[WARN] Empty protein sequences skipped after cleaning: {skipped_empty}")

    model = load_local_esmc300m(
        weight_path=esmc_weight_file,
        device=device,
        use_flash_attn=FIXED_ESMC_USE_FLASH_ATTN,
    )

    rows: List[List[Any]] = []
    failed_records: List[Tuple[str, str, str]] = []

    for index, record in enumerate(records, start=1):
        try:
            embedding = mean_pool_esmc_embedding(model, record.seq)
        except Exception as exc:
            error_message = str(exc).replace("\n", " ")
            failed_records.append((record.seq_id, record.source_file, error_message))
            print(f"[ERROR] ESM-C failed for {record.seq_id}: {error_message}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        rows.append([record.seq_id] + embedding.numpy().tolist())
        print(f"[INFO] Embedded ESM-C protein sequences: {index}/{len(records)}")

        del embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if failed_records:
        raise RuntimeError(
            f"ESM-C embedding failed for {len(failed_records)} sequence(s). "
            "The first error was: "
            f"{failed_records[0][0]} | {failed_records[0][2]}"
        )

    if not rows:
        raise RuntimeError("No ESM-C protein embeddings were generated.")

    embedding_dim = len(rows[0]) - 1
    return pd.DataFrame(rows, columns=["Seq_ID"] + [f"dim_{i}" for i in range(embedding_dim)])


def embed_proteins_esmc(
    input_path: str,
    device: Any,
    batch_size: int,
    esmc_weight_file: Path,
) -> Any:
    raw_records = load_fasta_records(input_path, "protein")
    return embed_protein_records_esmc(
        raw_records=raw_records,
        device=device,
        batch_size=batch_size,
        esmc_weight_file=esmc_weight_file,
    )


def split_sequence(seq: str, window_size: int, stride: int) -> List[str]:
    if len(seq) <= window_size:
        return [seq]

    windows: List[str] = []
    start = 0
    while start < len(seq):
        end = start + window_size
        window = seq[start:end]
        if window:
            windows.append(window)
        if end >= len(seq):
            break
        start += stride
    return windows


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_attention_mask(tokens: Any, tokenizer: Any) -> Any:
    input_ids = tokens["input_ids"]
    if "attention_mask" in tokens:
        return tokens["attention_mask"]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        return torch.ones_like(input_ids)
    return (input_ids != pad_token_id).long()


def check_ntv3_model_dir(model_dir: Path) -> None:
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "tokenization_ntv3.py",
        "modeling_ntv3_pretrained.py",
        "configuration_ntv3_pretrained.py",
    ]

    if not model_dir.exists():
        raise FileNotFoundError(f"Local NTv3 model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"NTv3 model path is not a directory: {model_dir}")

    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        missing_text = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing required NTv3 local file(s) in {model_dir}:\n{missing_text}\n\n"
            "For offline use, copy the NTv3 custom Python files into the NTv3_100M_pre directory "
            "and make sure config.json auto_map points to local references."
        )


def load_local_ntv3_model(model_dir: Path, device: Any) -> Tuple[Any, Any]:
    if AutoTokenizer is None or AutoModelForMaskedLM is None:
        raise ImportError("Package 'transformers' is required for NTv3 embedding. Install it with: pip install transformers")

    check_ntv3_model_dir(model_dir)

    # Some NTv3 custom modeling files use absolute local imports such as
    # 'configuration_ntv3_pretrained'. Adding the model directory to sys.path
    # keeps local offline loading robust without changing the command line.
    model_dir_resolved = model_dir.resolve()
    model_dir_text = str(model_dir_resolved)
    if model_dir_text not in sys.path:
        sys.path.insert(0, model_dir_text)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    print(f"[INFO] Loading local NTv3 model from: {model_dir_resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir_resolved),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_dir_resolved),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


def embed_ntv3_windows(
    windows: List[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int,
) -> Any:
    all_embeddings: List[Any] = []

    for start in range(0, len(windows), batch_size):
        batch_seqs = windows[start:start + batch_size]
        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=128,
            return_attention_mask=True,
            add_special_tokens=False,
        )

        attention_mask = build_attention_mask(tokens, tokenizer)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            window_embeddings = mean_pool(last_hidden, attention_mask)

        all_embeddings.append(window_embeddings.detach().cpu().numpy())

        del outputs, last_hidden, window_embeddings, tokens, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


def embed_nucleotide_records_ntv3(
    records: List[FastaRecord],
    device: Any,
    batch_size: int,
    nt_model_dir: Path,
    window_size: int = FIXED_NTV3_WINDOW_SIZE,
    stride: int = FIXED_NTV3_STRIDE,
) -> Any:
    if not records:
        raise ValueError("No nucleotide records were provided for embedding.")

    cleaned_records: List[FastaRecord] = []
    skipped_empty = 0
    for record in records:
        cleaned = clean_nucleotide_sequence(record.seq)
        if cleaned:
            cleaned_records.append(FastaRecord(seq_id=record.seq_id, seq=cleaned, source_file=record.source_file))
        else:
            skipped_empty += 1

    if not cleaned_records:
        raise ValueError("No valid nucleotide sequences remained after cleaning.")

    print(f"[INFO] Using device for NTv3: {device}")
    print(f"[INFO] NTv3 window size: {window_size}")
    print(f"[INFO] NTv3 stride: {stride}")
    print(f"[INFO] Nucleotide sequences to embed: {len(cleaned_records)}")
    if skipped_empty:
        print(f"[WARN] Empty nucleotide sequences skipped after cleaning: {skipped_empty}")

    tokenizer, model = load_local_ntv3_model(nt_model_dir, device)

    rows: List[List[Any]] = []
    for index, record in enumerate(cleaned_records, start=1):
        windows = split_sequence(record.seq, window_size=window_size, stride=stride)
        window_embeddings = embed_ntv3_windows(
            windows=windows,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        seq_embedding = window_embeddings.mean(axis=0)
        rows.append([record.seq_id] + seq_embedding.tolist())

        print(
            f"[INFO] Embedded NTv3 nucleotide sequence {index}/{len(cleaned_records)} "
            f"(length={len(record.seq)}, windows={len(windows)})"
        )

    if not rows:
        raise RuntimeError("No NTv3 nucleotide embeddings were generated.")

    embedding_dim = len(rows[0]) - 1
    return pd.DataFrame(rows, columns=["Seq_ID"] + [f"emb_{i}" for i in range(embedding_dim)])


def embed_nucleotides_ntv3(
    input_path: str,
    device: Any,
    batch_size: int,
    nt_model_dir: Path,
    window_size: int = FIXED_NTV3_WINDOW_SIZE,
    stride: int = FIXED_NTV3_STRIDE,
) -> Any:
    records = load_fasta_records(input_path, "nucleotide")
    return embed_nucleotide_records_ntv3(
        records=records,
        device=device,
        batch_size=batch_size,
        nt_model_dir=nt_model_dir,
        window_size=window_size,
        stride=stride,
    )


def normalize_match_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.(fa|fasta|faa|fna|fas|fsa)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([._-])(cds|nt|nucleotide|protein|aa)$", "", text, flags=re.IGNORECASE)
    return text.lower()


def make_concat_embedding(protein_df: Any, nucleotide_df: Any, intermediate_dir: Path) -> Any:
    p = protein_df.copy()
    n = nucleotide_df.copy()
    p["__match_id__"] = p["Seq_ID"].map(normalize_match_id)
    n["__match_id__"] = n["Seq_ID"].map(normalize_match_id)

    duplicated_p = p[p["__match_id__"].duplicated(keep=False)][["Seq_ID", "__match_id__"]]
    duplicated_n = n[n["__match_id__"].duplicated(keep=False)][["Seq_ID", "__match_id__"]]
    if not duplicated_p.empty:
        duplicated_p.to_csv(intermediate_dir / "concat_duplicated_protein_ids.csv", index=False)
        warnings.warn("Duplicated protein IDs after normalization. The first occurrence will be used.")
    if not duplicated_n.empty:
        duplicated_n.to_csv(intermediate_dir / "concat_duplicated_nucleotide_ids.csv", index=False)
        warnings.warn("Duplicated nucleotide IDs after normalization. The first occurrence will be used.")

    p = p.drop_duplicates(subset=["__match_id__"], keep="first")
    n = n.drop_duplicates(subset=["__match_id__"], keep="first")

    merged = p.merge(n, on="__match_id__", how="inner", suffixes=("_protein", "_nucleotide"))
    if merged.empty:
        unmatched_p = p[["Seq_ID", "__match_id__"]]
        unmatched_n = n[["Seq_ID", "__match_id__"]]
        unmatched_p_path = intermediate_dir / "concat_unmatched_protein_ids.csv"
        unmatched_n_path = intermediate_dir / "concat_unmatched_nucleotide_ids.csv"
        unmatched_p.to_csv(unmatched_p_path, index=False)
        unmatched_n.to_csv(unmatched_n_path, index=False)
        raise ValueError(
            "No protein/nucleotide records matched after ID normalization. "
            f"Check IDs in: {unmatched_p_path} and {unmatched_n_path}"
        )

    p_feature_cols = [c for c in protein_df.columns if c != "Seq_ID"]
    n_feature_cols = [c for c in nucleotide_df.columns if c != "Seq_ID"]

    # Build concat embedding columns in one dictionary and create the DataFrame once.
    # This avoids pandas PerformanceWarning caused by repeatedly inserting
    # hundreds or thousands of columns into the same DataFrame.
    output_columns: Dict[str, Any] = {
        "Seq_ID": merged["Seq_ID_protein"].astype(str).values
    }

    for col in p_feature_cols:
        merged_col = col if col in merged.columns else f"{col}_protein"
        if merged_col not in merged.columns:
            raise KeyError(f"Protein feature column not found after merge: {col}")
        output_columns[f"p_{col}"] = (
            pd.to_numeric(merged[merged_col], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )

    for col in n_feature_cols:
        merged_col = col if col in merged.columns else f"{col}_nucleotide"
        if merged_col not in merged.columns:
            raise KeyError(f"Nucleotide feature column not found after merge: {col}")
        output_columns[f"n_{col}"] = (
            pd.to_numeric(merged[merged_col], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )

    output = pd.DataFrame(output_columns).copy()

    p_keys = set(p["__match_id__"])
    n_keys = set(n["__match_id__"])
    unmatched_p = p[~p["__match_id__"].isin(n_keys)][["Seq_ID", "__match_id__"]]
    unmatched_n = n[~n["__match_id__"].isin(p_keys)][["Seq_ID", "__match_id__"]]
    if not unmatched_p.empty or not unmatched_n.empty:
        unmatched_p.to_csv(intermediate_dir / "concat_unmatched_protein_ids.csv", index=False)
        unmatched_n.to_csv(intermediate_dir / "concat_unmatched_nucleotide_ids.csv", index=False)
        print(f"[WARN] Unmatched protein records: {len(unmatched_p)}")
        print(f"[WARN] Unmatched nucleotide records: {len(unmatched_n)}")

    print(f"[INFO] Matched records for concat embedding: {len(output)}")
    return output


def normalize_embedding_kind(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"p", "protein"}:
        return "protein"
    if text in {"n", "nt", "nucleotide"}:
        return "nucleotide"
    if text in {"pn", "concat", "concet", "concatenated", "protein_nucleotide"}:
        return "concat"
    raise ValueError(f"Unsupported embedding type: {value}")


def natural_k_sort_key(k_name: str) -> Tuple[int, str]:
    match = re.search(r"K\s*(\d+)", str(k_name), flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), str(k_name)
    return 10**9, str(k_name)


def load_final_models(model_file: Path, embedding_kind: str) -> List[LoadedModel]:
    bundle = joblib_load_with_compact_sklearn_warning(model_file, "RF model bundle")
    if not isinstance(bundle, dict):
        raise ValueError(f"Model bundle must be a dictionary: {model_file}")

    models: List[LoadedModel] = []
    errors: List[str] = []

    for model_key, payload in bundle.items():
        try:
            if not isinstance(payload, dict):
                raise ValueError("model payload is not a dictionary")
            if "model" not in payload:
                raise ValueError("model payload has no 'model' field")

            kind = normalize_embedding_kind(payload.get("embedding_kind", payload.get("embedding_tag", "")))
            if kind != embedding_kind:
                continue

            k_name = str(payload.get("k", ""))
            if not k_name:
                match = re.search(r"K\d+", str(model_key), flags=re.IGNORECASE)
                k_name = match.group(0).upper() if match else str(model_key)

            feature_cols = payload.get("feature_cols", []) or []
            models.append(
                LoadedModel(
                    model_key=str(model_key),
                    model=payload["model"],
                    k_name=k_name,
                    embedding_kind=kind,
                    embedding_name=str(payload.get("embedding_name", model_key)),
                    feature_cols=[str(x) for x in feature_cols],
                    threshold=float(payload.get("threshold", 0.5)),
                )
            )
        except Exception as exc:
            errors.append(f"{model_key}: {exc}")

    models = sorted(models, key=lambda item: natural_k_sort_key(item.k_name))
    if not models:
        message = f"No {embedding_kind} models were found in {model_file}"
        if errors:
            message += "\nModel loading errors:\n" + "\n".join(errors[:20])
        raise FileNotFoundError(message)

    print(f"[INFO] Loaded {len(models)} {embedding_kind} models from {model_file}")
    return models


def load_platt_models(platt_file: Path) -> Dict:
    if platt_file is None:
        raise FileNotFoundError("Platt calibration is required, but no -pm/--platt-model file was provided.")
    try:
        models = joblib_load_with_compact_sklearn_warning(platt_file, "Platt model bundle")
    except Exception as exc:
        raise RuntimeError(f"Failed to load required Platt models from {platt_file}: {exc}") from exc
    if not isinstance(models, dict):
        raise ValueError(f"Required Platt model file is not a dictionary: {platt_file}")
    if not models:
        raise ValueError(f"Required Platt model dictionary is empty: {platt_file}")
    print(f"[INFO] Loaded required Platt models from {platt_file}")
    return models


def find_platt_model(platt_models: Dict, embedding_name: str, k_name: str):
    candidate_keys = [
        (embedding_name, k_name),
        (str(embedding_name), str(k_name)),
        (str(k_name), str(embedding_name)),
        f"{embedding_name}|{k_name}",
        f"{embedding_name}::{k_name}",
        f"{embedding_name}_{k_name}",
    ]
    for key in candidate_keys:
        if key in platt_models:
            return platt_models[key]
    return None


def get_numeric_feature_columns(df: Any, id_col: str) -> List[str]:
    excluded = {id_col, "__source_embedding_file__"}
    feature_cols: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            feature_cols.append(str(col))
    return feature_cols


def resolve_feature_matrix(df: Any, id_col: str, loaded_model: LoadedModel) -> Any:
    """Build the model input matrix using numeric columns in their current order.

    This intentionally ignores saved feature column names in the model bundle.
    It only checks the expected feature count when that information is available.
    """
    numeric_cols = get_numeric_feature_columns(df, id_col)
    if not numeric_cols:
        raise ValueError("No numeric embedding feature columns were found")

    x = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    expected_n = getattr(loaded_model.model, "n_features_in_", None)
    if expected_n is None and loaded_model.feature_cols:
        expected_n = len(loaded_model.feature_cols)

    if expected_n is not None:
        expected_n = int(expected_n)
        if x.shape[1] < expected_n:
            raise ValueError(
                f"{loaded_model.model_key}: model expects {expected_n} features, "
                f"but input has only {x.shape[1]} numeric features"
            )
        if x.shape[1] > expected_n:
            x = x[:, :expected_n]

    return x


def predict_positive_score(model: Any, x: Any) -> Any:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", []))
        pos_idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, pos_idx].astype(float)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x).astype(float)
        return 1.0 / (1.0 + np.exp(-decision))
    return np.asarray(model.predict(x), dtype=float)


def make_platt_model_compatible(platt_model: Any) -> Any:
    """Patch small sklearn-version compatibility gaps in old saved Platt models.

    Some LogisticRegression objects saved with older scikit-learn versions may miss
    attributes that newer scikit-learn versions expect during predict_proba().
    This does not refit or change coefficients; it only restores default metadata.
    """
    if platt_model is None:
        return None

    if platt_model.__class__.__name__ == "LogisticRegression":
        if not hasattr(platt_model, "multi_class"):
            setattr(platt_model, "multi_class", "auto")
        if not hasattr(platt_model, "n_jobs"):
            setattr(platt_model, "n_jobs", None)

    return platt_model


def compute_error_probability(score: Any, pred_label: Any, platt_model: Any, model_label: str) -> Any:
    if platt_model is None:
        raise ValueError(f"Platt calibration is required, but no Platt model was found for {model_label}.")

    try:
        platt_model = make_platt_model_compatible(platt_model)
        if not hasattr(platt_model, "predict_proba"):
            raise AttributeError("Platt model has no predict_proba method")
        p_pos = platt_model.predict_proba(score.reshape(-1, 1))[:, 1]
    except Exception as exc:
        raise RuntimeError(
            f"Required Platt calibration failed for {model_label}. "
            f"Prediction stopped because raw RF probability fallback is disabled. Reason: {exc}"
        ) from exc

    p_pos = np.clip(np.asarray(p_pos, dtype=float), 0.0, 1.0)
    p_neg = 1.0 - p_pos
    error_prob = np.where(pred_label == 1, p_neg, p_pos)
    return np.round(error_prob.astype(float), 6)


def build_prediction_table(
    embedding_df: Any,
    id_col: str,
    models: List[LoadedModel],
    platt_models: Dict,
) -> Any:
    # Build all columns in a dictionary and create the DataFrame once.
    # This avoids pandas PerformanceWarning caused by repeatedly inserting
    # hundreds of columns into the same DataFrame inside the model loop.
    output_columns: Dict[str, Any] = {
        "Seq_ID": embedding_df[id_col].astype(str).values
    }

    for loaded_model in models:
        k_name = str(loaded_model.k_name)
        x = resolve_feature_matrix(embedding_df, id_col, loaded_model)
        score = predict_positive_score(loaded_model.model, x)
        pred_label = (score >= loaded_model.threshold).astype(int)
        pred_text = np.where(pred_label == 1, "positive", "negative")

        platt_model = find_platt_model(
            platt_models=platt_models,
            embedding_name=loaded_model.embedding_name,
            k_name=loaded_model.k_name,
        )
        model_label = f"{loaded_model.embedding_name} / {loaded_model.k_name}"
        if platt_model is None:
            raise ValueError(
                "Platt calibration is required, but no matching Platt model was found for "
                f"{model_label}. Check keys in the -pm/--platt-model bundle."
            )
        error_prob = compute_error_probability(score, pred_label, platt_model, model_label)

        output_columns[f"{k_name}_score"] = np.round(score, 6)
        output_columns[f"{k_name}_prediction"] = pred_text
        output_columns[f"{k_name}_error_probability"] = error_prob

    return pd.DataFrame(output_columns).copy()




def is_positive_prediction(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"positive", "pos", "1", "true", "yes", "y", "正", "阳性"}


def parse_probability(value: object) -> float:
    if value is None or pd.isna(value):
        return float("inf")

    try:
        return float(value)
    except (TypeError, ValueError):
        matches = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", str(value))
        if not matches:
            return float("inf")
        return float(matches[-1])


def build_positive_ktype_rank_table(prediction_df: Any) -> Any:
    """Rank positive K-type predictions by ascending error probability.

    Output columns:
        ID
        K_type_rank

    K_type_rank contains semicolon-separated K types, for example:
        K64;K2;K1
    If a sequence has no positive K-type prediction, K_type_rank is empty.
    """
    if prediction_df.empty:
        return pd.DataFrame(columns=["ID", "K_type_rank"])

    id_col = "Seq_ID" if "Seq_ID" in prediction_df.columns else prediction_df.columns[0]
    prediction_cols = [col for col in prediction_df.columns if str(col).endswith("_prediction")]

    rows: List[Dict[str, str]] = []
    for _, row in prediction_df.iterrows():
        positive_items: List[Tuple[float, str]] = []

        for pred_col in prediction_cols:
            k_name = str(pred_col)[: -len("_prediction")]
            error_col = f"{k_name}_error_probability"
            if error_col not in prediction_df.columns:
                continue
            if not is_positive_prediction(row[pred_col]):
                continue

            error_probability = parse_probability(row[error_col])
            positive_items.append((error_probability, k_name))

        positive_items.sort(key=lambda item: (item[0], natural_k_sort_key(item[1])))
        rows.append(
            {
                "ID": str(row[id_col]),
                "K_type_rank": ";".join(k_name for _, k_name in positive_items),
            }
        )

    return pd.DataFrame(rows, columns=["ID", "K_type_rank"])

def write_table(df: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    try:
        validate_basic_args(args)
        import_runtime_modules()
        device = choose_device(None)
        rf_model_file = Path(args.ktype_model).resolve()
        platt_file = Path(args.platt_model).resolve()
        print(f"[INFO] K-type model bundle: {rf_model_file}")
        print(f"[INFO] Platt model bundle: {platt_file}")
        esmc_weight_file = discover_esmc300m_weight_file() if args.protein is not None else None
        nt_model_dir = discover_ntv3_model_dir() if args.nucleotide is not None else None
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    rank_output_path = Path(args.rank_output)
    full_output_path = Path(args.full_output) if args.full_output is not None else None
    output_base_path = full_output_path if full_output_path is not None else rank_output_path
    intermediate_dir = output_base_path.parent / f"{output_base_path.stem}_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    protein_df = None
    nucleotide_df = None
    paired_protein_records = None
    paired_nucleotide_records = None

    if args.protein is not None and args.nucleotide is not None:
        try:
            file_pairs, unmatched_pairs = build_file_pairs(args.protein, args.nucleotide, args.csv)
            print_ready_file_pairs(file_pairs)
            print_unmatched_file_pairs(unmatched_pairs)
            write_selected_pairs(file_pairs, intermediate_dir / "selected_pn_pairs.csv")
            write_unmatched_file_pairs(unmatched_pairs, intermediate_dir / "unmatched_pn_pairs.csv")
            if not file_pairs:
                raise ValueError(
                    "No protein/nucleotide file pairs were matched. "
                    f"Unmatched records were saved to: {intermediate_dir / 'unmatched_pn_pairs.csv'}"
                )
            paired_protein_records, paired_nucleotide_records = load_paired_fasta_records(file_pairs)
        except Exception as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)

    if args.protein is not None:
        if esmc_weight_file is None:
            raise RuntimeError("Protein input was provided, but the ESM-C 300M weight file was not resolved.")
        if paired_protein_records is not None:
            protein_df = embed_protein_records_esmc(
                raw_records=paired_protein_records,
                device=device,
                batch_size=FIXED_PROTEIN_BATCH_SIZE,
                esmc_weight_file=esmc_weight_file,
            )
        else:
            protein_df = embed_proteins_esmc(
                input_path=args.protein,
                device=device,
                batch_size=FIXED_PROTEIN_BATCH_SIZE,
                esmc_weight_file=esmc_weight_file,
            )
        protein_embedding_path = intermediate_dir / "protein_embedding.csv"
        protein_df.to_csv(protein_embedding_path, index=False)
        print(f"[INFO] Protein embedding saved to: {protein_embedding_path}")

    if args.nucleotide is not None:
        if nt_model_dir is None:
            raise RuntimeError("Nucleotide input was provided, but the NTv3 100M model directory was not resolved.")
        if paired_nucleotide_records is not None:
            nucleotide_df = embed_nucleotide_records_ntv3(
                records=paired_nucleotide_records,
                device=device,
                batch_size=FIXED_NUCLEOTIDE_BATCH_SIZE,
                nt_model_dir=nt_model_dir,
                window_size=FIXED_NTV3_WINDOW_SIZE,
                stride=FIXED_NTV3_STRIDE,
            )
        else:
            nucleotide_df = embed_nucleotides_ntv3(
                input_path=args.nucleotide,
                device=device,
                batch_size=FIXED_NUCLEOTIDE_BATCH_SIZE,
                nt_model_dir=nt_model_dir,
                window_size=FIXED_NTV3_WINDOW_SIZE,
                stride=FIXED_NTV3_STRIDE,
            )
        nucleotide_embedding_path = intermediate_dir / "nucleotide_embedding.csv"
        nucleotide_df.to_csv(nucleotide_embedding_path, index=False)
        print(f"[INFO] Nucleotide embedding saved to: {nucleotide_embedding_path}")

    if protein_df is not None and nucleotide_df is not None:
        embedding_kind = "concat"
        embedding_df = make_concat_embedding(protein_df, nucleotide_df, intermediate_dir)
        concat_embedding_path = intermediate_dir / "concat_embedding.csv"
        embedding_df.to_csv(concat_embedding_path, index=False)
        print(f"[INFO] Concat embedding saved to: {concat_embedding_path}")
    elif protein_df is not None:
        embedding_kind = "protein"
        embedding_df = protein_df
    else:
        embedding_kind = "nucleotide"
        embedding_df = nucleotide_df

    try:
        models = load_final_models(model_file=rf_model_file, embedding_kind=embedding_kind)
        platt_models = load_platt_models(platt_file)
        prediction_df = build_prediction_table(
            embedding_df=embedding_df,
            id_col="Seq_ID",
            models=models,
            platt_models=platt_models,
        )
        if full_output_path is not None:
            write_table(prediction_df, full_output_path)

        rank_df = build_positive_ktype_rank_table(prediction_df)
        write_table(rank_df, rank_output_path)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[DONE] Prediction mode: {embedding_kind}")
    print(f"[DONE] Input records predicted: {len(embedding_df)}")
    print(f"[DONE] Models used: {len(models)}")
    if full_output_path is not None:
        print(f"[DONE] Full prediction output saved to: {full_output_path}")
    else:
        print("[DONE] Full prediction output was not requested.")
    print(f"[DONE] Positive K-type rank output saved to: {rank_output_path}")


if __name__ == "__main__":
    main()
