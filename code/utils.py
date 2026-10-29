"""Shared helpers for the knights vs bishops project.

Used by every part of the project. Covers:

* the starting positions (from positions.py: the standard position with
  minor pieces or pawns removed);
* engine manifests (``engines/<name>/engine.toml``) and SHA-256 checks
  against ``engines/engines.lock.json``;
* opening a UCI engine with a fixed, recorded set of options;
* platform metadata recorded with every result;
* metric transforms: WDL -> expected score -> Elo, and convergence measures;
* small IO helpers (atomic Parquet writes, TOML loading).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import chess
import chess.engine

log = logging.getLogger("kvb")

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = REPO_ROOT / "engines"
CONFIG_DIR = REPO_ROOT / "config"
LOCK_FILE = ENGINES_DIR / "engines.lock.json"

# python-chess sets these itself (UCI_Chess960 from the board, MultiPV from
# the analysis call). Configuring them by hand raises an error.
MANAGED_OPTIONS = {"uci_chess960", "multipv", "ponder", "uci_variant"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Log to stdout with timestamps (readable in SLURM output files)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Starting positions (defined in positions.py; re-exported here)
# ---------------------------------------------------------------------------

from positions import (  # noqa: E402  (kept next to the other helpers)
    START_ID,
    Position,
    all_positions,
    get_position,
    parse_position_spec,
    position_id_from_fen,
)


# ---------------------------------------------------------------------------
# Engine manifests
# ---------------------------------------------------------------------------

class EngineSetupError(RuntimeError):
    """An engine is missing, unverified, or lacks a required option."""


def platform_key() -> str:
    """'linux', 'windows' or 'darwin': the key used in engine manifests."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


@dataclass
class EngineConfig:
    """An engine's manifest, resolved for the current platform."""

    key: str                      # folder name, e.g. "stockfish"
    folder: Path
    name: str
    version: str
    display_name: str             # label in figures and fastchess, e.g. "Bobby Bischer"
    search_type: str
    training_lineage: str
    binary: Path
    network: Path | None
    network_option: str | None
    uci_options: dict[str, Any]
    optional_uci_options: dict[str, Any]
    environment: dict[str, str]
    smoke_nodes: int
    raw: dict[str, Any] = field(repr=False)


def list_engines() -> list[str]:
    """Engine keys: every folder in engines/ that has an engine.toml."""
    return sorted(p.parent.name for p in ENGINES_DIR.glob("*/engine.toml"))


def load_engine_config(key: str) -> EngineConfig:
    """Load ``engines/<key>/engine.toml`` and resolve paths for this platform."""
    folder = ENGINES_DIR / key
    manifest = folder / "engine.toml"
    if not manifest.exists():
        raise EngineSetupError(
            f"no manifest at {manifest}; known engines: {', '.join(list_engines())}"
        )
    data = load_toml(manifest)
    info = data["engine"]

    binaries = data.get("binary", {})
    rel = binaries.get(platform_key())
    if not rel:
        raise EngineSetupError(f"{key}: no binary path for platform '{platform_key()}'")

    net = data.get("network", {})
    network = folder / net["file"] if net.get("file") else None

    return EngineConfig(
        key=key,
        folder=folder,
        name=info["name"],
        version=str(info["version"]),
        display_name=info.get("display_name", info["name"]),
        search_type=info.get("search_type", ""),
        training_lineage=info.get("training_lineage", ""),
        binary=folder / rel,  # not resolved: keep symlinks pointing into bin/
        network=network,
        network_option=net.get("uci_option"),
        # Options for every platform, plus any for this platform only
        # (e.g. a GPU backend on the Linux cluster).
        uci_options={**data.get("uci_options", {}),
                     **data.get("platform_uci_options", {}).get(platform_key(), {})},
        optional_uci_options=dict(data.get("optional_uci_options", {})),
        environment={k: str(v) for k, v in data.get("environment", {}).items()},
        smoke_nodes=int(info.get("smoke_nodes", 5000)),
        raw=data,
    )


# ---------------------------------------------------------------------------
# File hashes and the lock file
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_lock() -> dict[str, Any]:
    if not LOCK_FILE.exists():
        return {}
    data = json.loads(LOCK_FILE.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def save_lock(lock: dict[str, Any]) -> None:
    payload = {
        "_comment": (
            "SHA-256 of each engine binary and network, per platform. "
            "Written by `python code/setup_engines.py record`; checked before every run."
        )
    }
    payload.update(dict(sorted(lock.items())))
    LOCK_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def engine_fingerprint(cfg: EngineConfig) -> dict[str, Any]:
    """Hashes of the files that determine an engine's output."""
    if not cfg.binary.exists():
        raise EngineSetupError(
            f"{cfg.key}: binary not found at {cfg.binary} (see engines/{cfg.key}/README.md)"
        )
    fp: dict[str, Any] = {"binary_sha256": sha256_file(cfg.binary), "network_sha256": None}
    if cfg.network is not None:
        if not cfg.network.exists():
            raise EngineSetupError(f"{cfg.key}: network file not found at {cfg.network}")
        fp["network_sha256"] = sha256_file(cfg.network)
    return fp


def verify_engine_files(cfg: EngineConfig, strict: bool = True) -> dict[str, Any]:
    """Compare the engine's files with the lock file.

    A mismatch always raises. A missing lock entry raises when ``strict``
    (production runs) and only warns otherwise (development).
    """
    fp = engine_fingerprint(cfg)
    entry = load_lock().get(cfg.key, {}).get(platform_key())
    if entry is None:
        msg = (
            f"{cfg.key}: no lock entry for platform '{platform_key()}'. "
            "Run `python code/setup_engines.py record` once the engine is installed."
        )
        if strict:
            raise EngineSetupError(msg)
        log.warning(msg)
        return fp
    for item in ("binary_sha256", "network_sha256"):
        if entry.get(item) != fp[item]:
            raise EngineSetupError(
                f"{cfg.key}: {item} does not match the lock file "
                f"(expected {entry.get(item)}, found {fp[item]})"
            )
    return fp


# ---------------------------------------------------------------------------
# Opening engines
# ---------------------------------------------------------------------------

def apply_options(engine: chess.engine.SimpleEngine, cfg: EngineConfig) -> dict[str, Any]:
    """Set the manifest's UCI options and return what was applied.

    Required options must exist; optional ones are set only when the engine
    offers them.
    """
    applied: dict[str, Any] = {}
    for name, value in cfg.uci_options.items():
        if name.lower() in MANAGED_OPTIONS:
            raise EngineSetupError(f"{cfg.key}: '{name}' is set automatically; remove it")
        if name not in engine.options:
            raise EngineSetupError(f"{cfg.key}: engine has no UCI option '{name}'")
        applied[name] = value
    for name, value in cfg.optional_uci_options.items():
        if name in engine.options and name.lower() not in MANAGED_OPTIONS:
            applied[name] = value
    if cfg.network is not None:
        if not cfg.network_option or cfg.network_option not in engine.options:
            raise EngineSetupError(
                f"{cfg.key}: network option '{cfg.network_option}' not offered by engine"
            )
        applied[cfg.network_option] = str(cfg.network)
    engine.configure(applied)
    return applied


def open_engine(
    cfg: EngineConfig, strict_lock: bool = True, startup_timeout: float = 300.0,
    check_lock: bool = True,
) -> tuple[chess.engine.SimpleEngine, dict[str, Any]]:
    """Start an engine with the manifest's options.

    Returns the engine and a metadata dict (hashes, id, applied options and
    every option's default) to store alongside results.

    ``check_lock=False`` skips the lock-file comparison entirely. Only the
    setup tools use it (to inspect or re-record an engine whose files have
    changed); evaluation runs always check.
    """
    if check_lock:
        fingerprint = verify_engine_files(cfg, strict=strict_lock)
    else:
        fingerprint = engine_fingerprint(cfg)
    env = os.environ.copy()
    env.update(cfg.environment)
    engine = chess.engine.SimpleEngine.popen_uci(
        str(cfg.binary), timeout=startup_timeout, cwd=str(cfg.binary.parent), env=env
    )
    try:
        applied = apply_options(engine, cfg)
        engine.ping()  # waits for 'readyok', e.g. while a network loads
    except Exception:
        engine.quit()
        raise
    defaults = {name: opt.default for name, opt in engine.options.items()}
    id_name = engine.id.get("name", "")
    meta = {
        "engine": cfg.key,
        "engine_name": cfg.name,
        "engine_version": cfg.version,
        "engine_id_name": id_name,
        "version_matches_id": cfg.version.lower().lstrip("v") in id_name.lower(),
        **fingerprint,
        "uci_options_applied": applied,
        "uci_option_defaults": defaults,
        "environment": cfg.environment,
    }
    return engine, meta


# ---------------------------------------------------------------------------
# Platform metadata
# ---------------------------------------------------------------------------

def cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def gpu_model() -> str:
    """Name(s) of the NVIDIA GPU(s) visible to this process, or "" if none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return "; ".join(names)


def platform_info() -> dict[str, str]:
    return {
        "platform": platform_key(),
        "os": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu": cpu_model(),
        "gpu": gpu_model(),
        "python": platform.python_version(),
        "python_chess": chess.__version__,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_score_from_wdl(wins: float, draws: float, losses: float) -> float:
    """Expected score for the side the WDL refers to: (W + D/2) / (W + D + L)."""
    total = wins + draws + losses
    if total <= 0:
        return math.nan
    return (wins + 0.5 * draws) / total


def elo_from_expected(expected: float, eps: float = 1e-6) -> float:
    """Elo-equivalent advantage: 400 * log10(E / (1 - E)).

    This is a fixed transform of the expected score, so the two are one
    quantity on two scales, not independent results.
    """
    if expected is None or math.isnan(expected):
        return math.nan
    e = min(max(expected, eps), 1 - eps)
    return 400.0 * math.log10(e / (1.0 - e))


def expected_from_cp(cp: float, scale: float, offset: float = 0.0) -> float:
    """Logistic map from centipawns to expected score.

    ``scale`` and ``offset`` must be fitted (e.g. to self-play outcomes) for each
    engine that reports no WDL; there is no universal default.
    """
    return 1.0 / (1.0 + math.exp(-(cp - offset) / scale))


def total_variation(values: Sequence[float]) -> float:
    """Sum of absolute changes between successive budgets."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(sum(abs(b - a) for a, b in zip(vals, vals[1:])))


def tail_drift(values: Sequence[float], k: int = 3) -> float:
    """Largest gap between the top-budget value and the k budgets below it."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return math.nan
    top = vals[-1]
    return float(max(abs(v - top) for v in vals[-(k + 1):-1]))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_parquet_atomic(df, path: Path) -> None:
    """Write a DataFrame to Parquet via a temporary file, so an interrupted
    job never leaves a half-written result that looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
