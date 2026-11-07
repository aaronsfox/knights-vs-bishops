# Plans and design decisions

The plans are living documents in Claude (private to their owner):

- [Research plan](https://claude.ai/artifact/KuWfR4oCWbLk8yBMNjcq77): question,
  prior work, the three steps, tooling, compute budget, risks, paper outline.
- [Analysis plan](https://claude.ai/artifact/8Dbh5FjcsbvYZmvQxnzkhr): the
  three steps (human play, preference engines, removal matrix),
  their data, the shared statistical framework, and Questions 1-3.

Export a dated copy here (`analysis_plan_YYYY-MM-DD.md`) when the analysis is
locked, so the repository records the plan the results were produced under.

## Decisions the code depends on

| Decision | Where | Why |
| --- | --- | --- |
| Position ids name the removed squares, White's first (`c1-b8`) | `code/positions.py` | Readable in file names, tables and figures; no lookup table |
| Each position is pooled with its colour-flipped twin and scored from the focal side | `positions.py`, `analyse_removal.py` | In self-play, swapping engine copies doesn't move the material, so only the twin cancels White's edge |
| Focal side: more bishops, else more pawns, else the earlier-file removal | `positions.py` | One rule covers every design, including same-material square comparisons |
| Mirrored controls (own twin) report White's edge, not a focal score | `positions.py` | Gate 3: White's edge should be positive and similar across the standard position and every mirrored control |
| Opening lines generated from each position, balanced on expected score | `opening_book_generator.py` | Standard books don't apply with pieces missing; each cell keeps its imbalance |
| Key comparisons topped up as a second batch (`--batch b2`) | `selfplay.py` | Resumable; DEFF uses Kish's weighted cluster size for unequal batches |
| Engines play by time control, not a fixed node budget | `config/selfplay.toml` | A fixed node budget replays the same game for each line and colour (tested) |
| Stockfish 11 shift split half up, half down, in each phase's pawn units | `engines/stockfish11/patch_sf11.py` | Changes B - N without changing minor-piece material against pawns and rooks |
| Variants change the constants everywhere, not only in evaluation | `patch_sf11.py` | The engine believes its values in search (SEE, pruning) too |
| Pawn equivalents use each engine's own pawn-odds games | `analyse_removal.py` | Puts engines on one scale despite different WDL models; human games use their own pawn-up spells |
| The strength ladder (reduced-strength Stockfish, Maia-3) is dropped | plans | Human games stratified by rating answer the strength question directly |
| Tablebases only for endgames reached in human games | `tablebase.py` | The question is about decisions during play; sampled endgames add theory, not decisions |
