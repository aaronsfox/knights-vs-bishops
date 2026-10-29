"""Check, fingerprint and smoke-test the engines in engines/.

Subcommands
-----------
check   For each engine: is the binary (and network) present, does it match
        the lock file, does it start, and does it offer the required options?
record  Write the SHA-256 of each installed engine's files to
        engines/engines.lock.json for this platform. Run once after
        installing or upgrading an engine, then commit the lock file.
smoke   Run a short search on every starting position of the study and
        confirm the engine returns a legal move.

Examples
--------
    python code/setup_engines.py check
    python code/setup_engines.py record --engine stockfish
    python code/setup_engines.py smoke --engine caissa
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import chess
import chess.engine

from utils import (
    EngineSetupError,
    engine_fingerprint,
    get_position,
    list_engines,
    load_engine_config,
    load_lock,
    open_engine,
    parse_position_spec,
    platform_key,
    save_lock,
    setup_logging,
    utc_now,
)

log = logging.getLogger("kvb.setup")


def selected(args: argparse.Namespace) -> list[str]:
    return [args.engine] if args.engine else list_engines()


def cmd_check(args: argparse.Namespace) -> int:
    ok = True
    lock = load_lock()
    for key in selected(args):
        print(f"\n== {key}")
        try:
            cfg = load_engine_config(key)
            print(f"   manifest : {cfg.name} {cfg.version} ({cfg.search_type})"
                  + (f", shown as '{cfg.display_name}'" if cfg.display_name != cfg.name else ""))
            print(f"   binary   : {cfg.binary}")
            if cfg.network:
                print(f"   network  : {cfg.network}")
            fp = engine_fingerprint(cfg)
            entry = lock.get(key, {}).get(platform_key())
            if entry is None:
                print("   lock     : no entry for this platform (run `record`)")
            elif all(entry.get(k) == fp[k] for k in ("binary_sha256", "network_sha256")):
                print("   lock     : matches")
            else:
                print("   lock     : MISMATCH: files differ from the recorded hashes "
                      "(if you rebuilt or upgraded this engine on purpose, run `record`)")
                ok = False
            engine, meta = open_engine(cfg, check_lock=False)
            try:
                print(f"   id name  : {meta['engine_id_name']}")
                if not meta["version_matches_id"]:
                    print(f"   WARNING  : engine.toml pins version {cfg.version}; "
                          "the engine reports a different name")
                    ok = False
                print(f"   options  : {meta['uci_options_applied']}")
                missing = [n for n in cfg.optional_uci_options if n not in engine.options]
                if missing:
                    print(f"   optional options not offered: {missing}")
                if "UCI_ShowWDL" not in engine.options:
                    print("   note     : no UCI_ShowWDL; centipawns only (expected scores come from games)")
            finally:
                engine.quit()
        except (EngineSetupError, chess.engine.EngineError, OSError) as exc:
            print(f"   ERROR    : {exc}")
            ok = False
    return 0 if ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    lock = load_lock()
    status = 0
    for key in selected(args):
        try:
            cfg = load_engine_config(key)
            fp = engine_fingerprint(cfg)
            engine, meta = open_engine(cfg, check_lock=False)
            engine.quit()
        except (EngineSetupError, chess.engine.EngineError, OSError) as exc:
            log.error("%s: %s", key, exc)
            status = 1
            continue
        lock.setdefault(key, {})[platform_key()] = {
            "version": cfg.version,
            "id_name": meta["engine_id_name"],
            **fp,
            "recorded_utc": utc_now(),
        }
        if not meta["version_matches_id"]:
            log.warning("%s: engine reports '%s' but engine.toml pins version %s",
                        key, meta["engine_id_name"], cfg.version)
        log.info("%s: recorded %s", key, fp["binary_sha256"][:16])
    save_lock(lock)
    return status


def cmd_smoke(args: argparse.Namespace) -> int:
    status = 0
    for key in selected(args):
        cfg = load_engine_config(key)
        nodes = args.nodes or cfg.smoke_nodes
        try:
            engine, _ = open_engine(cfg, check_lock=False)
        except (EngineSetupError, chess.engine.EngineError, OSError) as exc:
            log.error("%s: %s", key, exc)
            status = 1
            continue
        failures = []
        pos_ids = parse_position_spec(args.positions)
        log.info("%s: smoke test on %d positions at %d nodes each", key, len(pos_ids), nodes)
        start = time.perf_counter()
        try:
            for n, pos_id in enumerate(pos_ids, 1):
                if n == 1 or n % 50 == 0 or n == len(pos_ids):
                    elapsed = time.perf_counter() - start
                    log.info("%s: position %d/%d (%.0fs elapsed)", key, n, len(pos_ids), elapsed)
                board = get_position(pos_id).board()
                try:
                    result = engine.play(board, chess.engine.Limit(nodes=nodes), game=object())
                    if result.move is None or not board.is_legal(result.move):
                        failures.append((pos_id, f"illegal or missing move {result.move}"))
                except chess.engine.EngineError as exc:
                    failures.append((pos_id, str(exc)))
        finally:
            engine.quit()
        if failures:
            status = 1
            log.error("%s: %d positions failed, e.g. %s", key, len(failures), failures[:3])
        else:
            log.info("%s: all positions returned legal moves at %d nodes", key, nodes)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, help_text in [
        ("check", cmd_check, "check installed engines"),
        ("record", cmd_record, "write file hashes to the lock file"),
        ("smoke", cmd_smoke, "short search on every starting position"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--engine", help="one engine (default: all)")
        p.set_defaults(func=func)
        if name == "smoke":
            p.add_argument("--positions", default="all")
            p.add_argument("--nodes", type=int, help="override the manifest's smoke_nodes")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
