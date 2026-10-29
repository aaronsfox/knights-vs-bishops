"""Strength ladder: games played by Maia-3 (human-like play), outside fastchess.

Maia-3 predicts the move a human of a given rating would play; it has no UCI
interface, so this runner plays the games itself: each position's book
lines, ``games_per_line`` games each with colours swapped in pairs, moves
**sampled** from the model's policy (taking the top move every time would
replay the same game). Games are adjudicated by Syzygy tablebases (if
configured) and by a Stockfish check (both sides' evaluations beyond the
resign threshold, or a draw by move cap), since a model without search may
not finish won endgames reliably.

The PGNs are written exactly where and how selfplay.py writes fastchess's
(raw/<player>/<position>.pgn.gz, book moves commented {book}, a Round and
Termination header), so ``selfplay.py collect`` and the analyses read them
unchanged.

Status: the Maia-3 policy (Maia3Policy) is a placeholder until its Python
interface (model loading, rating conditioning on own and opponent rating,
move probabilities) is checked against the Maia-3 repository. UciPolicy, a
UCI engine at a node budget, plays through the same code for testing.

    python code/maia_runner.py run --player maia3_1500 --positions key
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chess
import chess.engine
import chess.pgn
import chess.syzygy
import pandas as pd

from selfplay import book_path, load_book, output_dir, raw_path
from utils import (CONFIG_DIR, load_engine_config, load_toml, open_engine, parse_position_spec,
                   setup_logging)

log = logging.getLogger("kvb.maia")
DEFAULT_CONFIG = CONFIG_DIR / "maia.toml"


class Policy(Protocol):
    def choose(self, board: chess.Board, rng: random.Random) -> chess.Move: ...
    def close(self) -> None: ...


class Maia3Policy:
    """Maia-3 conditioned on a rating, sampling from its move probabilities.

    To implement once the maia3 package's interface is confirmed: load the
    model (5M, 23M or 79M), get move probabilities for ``board`` given the
    player's and the opponent's rating, then sample with ``temperature``."""

    def __init__(self, model: str, rating: int, opponent_rating: int, temperature: float):
        self.model, self.rating = model, rating
        self.opponent_rating, self.temperature = opponent_rating, temperature
        raise NotImplementedError(
            "Maia-3 policy not wired up yet: confirm the maia3 package's API (model loading, "
            "rating conditioning, move probabilities) and implement Maia3Policy.choose")

    def choose(self, board: chess.Board, rng: random.Random) -> chess.Move:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass


class UciPolicy:
    """A UCI engine at a fixed node budget (for testing the runner)."""

    def __init__(self, engine_key: str, nodes: int):
        self.engine, _ = open_engine(load_engine_config(engine_key), strict_lock=False)
        self.nodes = nodes

    def choose(self, board: chess.Board, rng: random.Random) -> chess.Move:
        return self.engine.play(board, chess.engine.Limit(nodes=self.nodes)).move

    def close(self) -> None:
        self.engine.quit()


@dataclass(frozen=True)
class Adjudication:
    engine: str
    nodes: int
    resign_cp: int = 1000
    resign_moves: int = 5
    max_moves: int = 200
    tablebases: str = ""
    tbpieces: int = 6


class Adjudicator:
    def __init__(self, a: Adjudication):
        self.a = a
        self.engine, _ = open_engine(load_engine_config(a.engine), strict_lock=False)
        self.tb = chess.syzygy.open_tablebase(a.tablebases) if a.tablebases else None

    def verdict(self, board: chess.Board, streak: list[int]) -> tuple[str, str] | None:
        """(result, reason) if the game should end now, else None."""
        if board.is_game_over(claim_draw=True):
            out = board.outcome(claim_draw=True)
            return out.result(), out.termination.name.lower().replace("_", " ")
        if board.fullmove_number > self.a.max_moves:
            return "1/2-1/2", "move cap"
        if self.tb and chess.popcount(board.occupied) <= self.a.tbpieces:
            try:
                wdl = self.tb.probe_wdl(board)
            except (KeyError, chess.syzygy.MissingTableError):
                wdl = None
            if wdl is not None:
                white = wdl if board.turn == chess.WHITE else -wdl
                return ("1-0" if white > 1 else "0-1" if white < -1 else "1/2-1/2"), "SyzygyTB"
        info = self.engine.analyse(board, chess.engine.Limit(nodes=self.a.nodes))
        cp = info["score"].white().score(mate_score=100_000)
        streak.append(cp)
        recent = streak[-2 * self.a.resign_moves:]
        if len(recent) == 2 * self.a.resign_moves:
            if all(c >= self.a.resign_cp for c in recent):
                return "1-0", "White wins by adjudication"
            if all(c <= -self.a.resign_cp for c in recent):
                return "0-1", "Black wins by adjudication"
        return None

    def close(self) -> None:
        self.engine.quit()
        if self.tb:
            self.tb.close()


def play_game(start_fen: str, book: list[str], white: Policy, black: Policy, adj: Adjudicator,
              rng: random.Random) -> tuple[chess.pgn.Game, str, str]:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.setup(board)
    node = game
    for uci in book:
        move = chess.Move.from_uci(uci)
        board.push(move)
        node = node.add_variation(move, comment="book")
    streak: list[int] = []
    while True:
        v = adj.verdict(board, streak)
        if v:
            result, reason = v
            break
        mover = white if board.turn == chess.WHITE else black
        move = mover.choose(board, rng)
        board.push(move)
        node = node.add_variation(move)
    node.comment = reason
    return game, result, reason


def cmd_run(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    p = config["players"][args.player]
    out_dir = output_dir(load_toml(Path(args.selfplay_config)), args.output_dir)
    book = load_book(book_path(load_toml(Path(args.selfplay_config))))
    games_per_line = int(args.games_per_line or config["match"]["games_per_line"])
    if p["policy"] == "maia3":
        def make(rating, opp):
            return Maia3Policy(p["model"], rating, opp, float(p.get("temperature", 1.0)))
        a = make(int(p["rating"]), int(p["rating"]))
        b = make(int(p["rating"]), int(p["rating"]))
    elif p["policy"] == "uci":
        a, b = UciPolicy(p["engine"], int(p["nodes"])), UciPolicy(p["engine"], int(p["nodes"]))
    else:
        raise SystemExit(f"unknown policy {p['policy']!r}")
    adj = Adjudicator(Adjudication(**config["adjudication"]))
    try:
        for pos_id in parse_position_spec(args.positions):
            dest = raw_path(out_dir, args.player, pos_id, args.batch)
            if dest.exists() and not args.overwrite:
                continue
            lines = book[book.position_id == pos_id].sort_values("line_index")
            rng = random.Random(f"{config['match']['seed']}:{args.player}:{pos_id}:{args.batch}")
            pgns, rnd = [], 0
            for _ in range(games_per_line // 2):
                for row in lines.itertuples():
                    rnd += 1
                    for swap in (False, True):
                        names = (f"{args.player}_b", f"{args.player}_a") if swap else \
                                (f"{args.player}_a", f"{args.player}_b")
                        w, bl = (b, a) if swap else (a, b)
                        game, result, reason = play_game(row.start_fen, row.moves_uci.split(), w, bl,
                                                         adj, rng)
                        game.headers.update({"Event": "Maia runner", "Round": str(rnd),
                                             "White": names[0], "Black": names[1], "Result": result,
                                             "Termination": "adjudication" if "adjudication" in reason
                                             or reason == "SyzygyTB" else "normal"})
                        pgns.append(str(game))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            with gzip.open(tmp, "wt") as fh:
                fh.write("\n\n".join(pgns) + "\n")
            os.replace(tmp, dest)
            log.info("%s: %d games", pos_id, len(pgns))
    finally:
        for x in (a, b):
            x.close()
        adj.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--selfplay-config", default=str(CONFIG_DIR / "selfplay.toml"),
                   help="for the book and output folder")
    r.add_argument("--output-dir")
    r.add_argument("--player", required=True, help="a player in [players] of the config")
    r.add_argument("--positions", default="key")
    r.add_argument("--games-per-line", type=int)
    r.add_argument("--batch", default="")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
