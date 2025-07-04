use std::collections::{BTreeSet, HashMap};
use std::fmt::{self, Display, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::{io, mem};

use clap::{Args, Parser, Subcommand};
use rand::seq::SliceRandom;
use rayon::prelude::*;

use board_game_traits::{Color, GameResult, Position as PositionTrait};
use chrono::{DateTime, Utc};
use pgn_traits::PgnPosition;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_rusqlite::{from_row, from_rows, to_params_named};
use tiltak::position::{self, ExpMove, Komi, Move, Position, Role};
use tiltak::search::{self, MctsSetting, MonteCarloTree};
use topaz_tak::board::{Board5, Board6};

use crate::gaelet::{analyze_puzzle_cataklysm, cataklysm_search, reserves_left_for_us};

const TILTAK_SHALLOW_NODES: u32 = 50_000;
const TILTAK_DEEP_NODES: u32 = 2_000_000;

const TOPAZ_FIRST_MOVE_NODES: usize = 10_000_000;
const TOPAZ_SECOND_MOVE_NODES: usize = 20_000_000;
const TOPAZ_AVOIDANCE_NODES: usize = 5_000_000;

mod export;
mod followups;
mod gaelet;
#[cfg(test)]
mod tests;

static NUM_GAMES_PROCESSED: AtomicU64 = AtomicU64::new(0);

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    #[arg(short, long)]
    size: usize,
    #[arg(long, default_value = "puzzles.db")]
    db_path: String,
    #[command(subcommand)]
    command: CliCommands,
}

#[derive(Subcommand)]
enum CliCommands {
    /// Find puzzles among positions imported from the Playtak database
    FindRootPuzzles(FindRootPuzzlesArgs),
    /// Find followups for puzzles that have already been identified
    FindFollowups,
    /// Gather together full tinue puzzles, and go through them manually
    FindFullPuzzles,
    FindRootGaelets,
    FindImmediateWins,
    ExtendTinuePuzzles,
    ShowPuzzle,
    ExportPuzzles(ExportPuzzlesArgs),
    FindFollowupsNew,
    FindGaelets,
    /// Analyze puzzles already in the database with Cataklysm, and interactively (i.e. manually) add them to the database
    AnalyzePuzzles,
    /// Analyze all games in the database with Cataklysm
    AnalyzeGames,
    /// Manually evaluate followup tinue puzzles
    EvaluateFollowupTinues,
    /// One-off command to retroactively backfill Cataklysm evaluations for candidate solutions for tinue puzzles
    BackfillCataklysmEval,
}

#[derive(Args)]
struct FindRootPuzzlesArgs {
    /// Path to the Playtak database file, to import games into the puzzles database. If not provided, only previously imported games will be analyzed.
    #[arg(short, long)]
    playtak_db_path: Option<String>,
}

#[derive(Args)]
struct ExportPuzzlesArgs {
    #[arg(long)]
    output_path: String,
}

fn main() {
    let cli_args = CliArgs::parse();
    let db_path = cli_args.db_path;
    match (cli_args.command, cli_args.size) {
        (CliCommands::FindRootPuzzles(args), 5) => main_sized::<5>(&args.playtak_db_path, &db_path),
        (CliCommands::FindRootPuzzles(args), 6) => main_sized::<6>(&args.playtak_db_path, &db_path),

        (CliCommands::FindFollowups, 5) => find_followups::<5>(&db_path),
        (CliCommands::FindFollowups, 6) => find_followups::<6>(&db_path),

        (CliCommands::FindFullPuzzles, 5) => find_full_puzzles::<5>(&db_path),
        (CliCommands::FindFullPuzzles, 6) => find_full_puzzles::<6>(&db_path),

        (CliCommands::FindRootGaelets, 5) => find_root_gaelets::<5>(&db_path),
        (CliCommands::FindRootGaelets, 6) => find_root_gaelets::<6>(&db_path),

        (CliCommands::FindImmediateWins, 5) => find_immediate_wins::<5>(&db_path),
        (CliCommands::FindImmediateWins, 6) => find_immediate_wins::<6>(&db_path),

        (CliCommands::ExtendTinuePuzzles, 5) => extend_tinue_puzzles::<5>(&db_path),
        (CliCommands::ExtendTinuePuzzles, 6) => extend_tinue_puzzles::<6>(&db_path),

        (CliCommands::ShowPuzzle, 5) => show_puzzle::<5>(&db_path),
        (CliCommands::ShowPuzzle, 6) => show_puzzle::<6>(&db_path),

        (CliCommands::ExportPuzzles(args), 5) => {
            export::export_puzzles::<5>(&db_path, &args.output_path)
        }
        (CliCommands::ExportPuzzles(args), 6) => {
            export::export_puzzles::<6>(&db_path, &args.output_path)
        }
        (CliCommands::FindFollowupsNew, 5) => followups::find_all_from_db::<5>(&db_path),
        (CliCommands::FindFollowupsNew, 6) => followups::find_all_from_db::<6>(&db_path),

        (CliCommands::FindGaelets, 5) => gaelet::find_potential_gaelet::<5>(&db_path),
        (CliCommands::FindGaelets, 6) => gaelet::find_potential_gaelet::<6>(&db_path),

        (CliCommands::AnalyzePuzzles, 5) => analyze_puzzles_cataklysm::<5>(&db_path),
        (CliCommands::AnalyzePuzzles, 6) => analyze_puzzles_cataklysm::<6>(&db_path),

        (CliCommands::AnalyzeGames, 5) => analyze_games_cataklysm::<5>(&db_path),
        (CliCommands::AnalyzeGames, 6) => analyze_games_cataklysm::<6>(&db_path),

        (CliCommands::EvaluateFollowupTinues, 5) => followups::evaluate_followups::<5>(&db_path),
        (CliCommands::EvaluateFollowupTinues, 6) => followups::evaluate_followups::<6>(&db_path),

        (CliCommands::BackfillCataklysmEval, 5) => {
            followups::backfill_cataklysm_evals::<5>(&db_path)
        }
        (CliCommands::BackfillCataklysmEval, 6) => {
            followups::backfill_cataklysm_evals::<6>(&db_path)
        }

        (_, s @ 7..) | (_, s @ 0..5) => panic!("Unsupported size: {}", s),
    }
}

#[derive(Debug)]
pub struct PuzzleRoot<const S: usize> {
    playtak_game_id: u32,
    tps: String,
    solution: Move<S>,
    tinue_length: usize,
}

impl<const S: usize> fmt::Display for PuzzleRoot<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tps {}, solution {}, length {}",
            self.tps, self.solution, self.tinue_length
        )
    }
}

fn deseralize_move<'de, D, const S: usize>(deserializer: D) -> Result<Move<S>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    Move::from_string(&s).map_err(serde::de::Error::custom)
}

pub fn analyze_games_cataklysm<const S: usize>(db_name: &str) {
    let mut games_conn = Connection::open(db_name).unwrap();
    let all_games: Vec<PlaytakGame> = read_all_games::<S>(&mut games_conn)
        .into_iter()
        .filter(|game| game.komi.half_komi() == 0 || game.komi.half_komi() == 4)
        .collect();
    let stats = Stats::default();

    println!("Found {} non-bot games", all_games.len());
    let start_time = Instant::now();

    all_games.into_par_iter()
        .for_each_init(
        || Connection::open(db_name).unwrap(),
        |conn, game| {
            let mut position: Position<S> = Position::start_position_with_komi(game.komi);
            for mv in game.notation.split_whitespace() {
                let mv = Move::from_string(mv).unwrap();
                assert!(position.move_is_legal(mv));
                position.do_move(mv);
                // Tinue is only possible after white has played S - 2 moves
                if position.half_moves_played() < (S - 2) * 2 {
                    continue;
                }

                let mut stmt = conn
                    .prepare("SELECT * FROM puzzles WHERE tps = ?1 AND tinue_length IS NOT NULL")
                    .unwrap();
                let mut rows = stmt.query([position.to_fen()]).unwrap();
                if rows.next().unwrap().is_some() {
                    // A topaz tinue has already been found, skip
                    continue;
                }

                // Search with 64 MiB tt
                let (eval, pv) = cataklysm_search(position.clone(), 7, &stats, 1 << 21);
                if eval.is_decisive()
                    && !eval.to_string().starts_with("loss")
                    && pv.split_whitespace().count() > 1
                {
                    let topaz_result = topaz_search_one_move::<S>(&position.to_fen(), &stats);
                    if topaz_result == Some(false) {
                        let tiltak_start_time = Instant::now();
                        let tiltak_result = tiltak_search(position.clone(), TILTAK_DEEP_NODES);
                        stats.tiltak_tinue.record(tiltak_start_time.elapsed());
                        let tiltak_delta = tiltak_result.score_first - tiltak_result.score_second;
                        if reserves_left_for_us(&position) > 4 || tiltak_result.score_first < 0.9 || tiltak_delta > 0.1 {
                            println!(
                                "Found decisive non-tinue eval \"{}\", {} reserves left, pv {} at # {}",
                                eval,
                                reserves_left_for_us(&position),
                                pv,
                                game.id
                            );
                            println!(
                                "Tiltak 1st: {:.3} {}, 2nd: {:.3} {}",
                                tiltak_result.score_first,
                                tiltak_result
                                    .pv_first
                                    .iter()
                                    .take(3)
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join(" "),
                                tiltak_result.score_second,
                                tiltak_result
                                    .pv_second
                                    .iter()
                                    .take(3)
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join(" "),
                            );
                            println!("{}", position.to_fen());
                        }
                    }
                }
            }
            let n = NUM_GAMES_PROCESSED.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 100 == 0 {
                println!(
                    "Processed {} games in {:.1}s",
                    n,
                    start_time.elapsed().as_secs_f64()
                );
                println!("{}", stats);
            }
        },
    )
}

pub fn analyze_puzzles_cataklysm<const S: usize>(db_name: &str) {
    let puzzles_conn = rusqlite::Connection::open(db_name).unwrap();

    let puzzle_roots: Vec<PuzzleRoot<S>> = puzzles_conn.prepare("SELECT puzzles.tps, puzzles.solution, puzzles.tinue_length, games.id FROM puzzles JOIN games ON puzzles.game_id = games.id
        WHERE games.size = ?1 AND puzzles.tinue_length NOT NULL AND (tiltak_0komi_second_move_eval < 0.7 OR tiltak_2komi_second_move_eval < 0.7)")
    .unwrap()
        .query([S])
        .unwrap()
        .mapped(|row| {
            Ok(PuzzleRoot {
                playtak_game_id: row.get::<_, u32>(3).unwrap(),
                tps: row.get(0).unwrap(),
                solution: Move::from_string(&row.get::<_, String>(1).unwrap()).unwrap(),
                tinue_length: row.get(2).unwrap(),
            })
        })
        .map(|row| row.unwrap())
        .collect();

    println!("Found {} puzzles roots", puzzle_roots.len());

    let num_wins = AtomicU64::new(0);
    let num_no_wins = AtomicU64::new(0);
    let start_time = Instant::now();

    puzzle_roots.into_par_iter().for_each(|puzzle| {
        let (eval, _pv) = analyze_puzzle_cataklysm(&puzzle);
        if eval.is_decisive() {
            num_wins.fetch_add(1, Ordering::Relaxed);
        } else {
            num_no_wins.fetch_add(1, Ordering::Relaxed);
        }
        let n = num_wins.load(Ordering::Relaxed) + num_no_wins.load(Ordering::Relaxed);
        if n % 100 == 0 {
            println!(
                "Processed {} puzzles, found {} wins, no wins {} in {:.1}s",
                n,
                num_wins.load(Ordering::Relaxed),
                num_no_wins.load(Ordering::Relaxed),
                start_time.elapsed().as_secs_f64()
            );
        }
    });
    println!(
        "Processed {} puzzles, found {} wins, no wins {} in {:.1}s",
        num_wins.load(Ordering::Relaxed) + num_no_wins.load(Ordering::Relaxed),
        num_wins.load(Ordering::Relaxed),
        num_no_wins.load(Ordering::Relaxed),
        start_time.elapsed().as_secs_f64()
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FinishedPuzzle {
    size: usize,
    half_komi: u32,
    root_tps: String,
    defender_start_move: String,
    solution: String,
    playtak_game_id: usize,
    player_white: String,
    player_black: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrontendPuzzle {
    size: usize,
    komi: String,
    #[serde(rename = "rootTPS")]
    root_tps: String,
    defender_start_move: String,
    solution: Vec<String>,
    target_time_seconds: u32,
    player_white: String,
    player_black: String,
    playtak_game_id: usize,
}

impl From<FinishedPuzzle> for FrontendPuzzle {
    fn from(puzzle: FinishedPuzzle) -> Self {
        let komi = Komi::from_half_komi(puzzle.half_komi as i8).unwrap();
        let mut solution: Vec<String> = puzzle
            .solution
            .split_whitespace()
            .map(ToString::to_string)
            .collect();
        // Puzzle solutions should always have an odd amount of moves
        // If the data database doesn't, the last move should allow any immediate win
        if solution.len() % 2 == 0 {
            solution.push("*".to_string());
        }
        FrontendPuzzle {
            size: puzzle.size,
            komi: komi.to_string(),
            root_tps: puzzle.root_tps,
            defender_start_move: puzzle.defender_start_move,
            solution,
            target_time_seconds: 120, // Default target time for puzzles
            player_white: puzzle.player_white,
            player_black: puzzle.player_black,
            playtak_game_id: puzzle.playtak_game_id,
        }
    }
}

fn read_random_puzzle(puzzles_conn: &Connection, size: usize) -> FinishedPuzzle {
    puzzles_conn
        .prepare("SELECT full_tinue_puzzles.*, games.player_white, games.player_black, games.size, games.komi as half_komi FROM full_tinue_puzzles JOIN games on full_tinue_puzzles.playtak_game_id = games.id WHERE games.size = ?1 ORDER BY RANDOM() LIMIT 1")
        .unwrap()
        .query_and_then([size], from_row::<FinishedPuzzle>)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()[0].clone()
}

fn show_puzzle<const S: usize>(db_path: &str) {
    let puzzles_conn = Connection::open(db_path).unwrap();

    let puzzle: FinishedPuzzle = read_random_puzzle(&puzzles_conn, S);

    let frontend_puzzle: FrontendPuzzle = puzzle.clone().into();
    println!(
        "Puzzle JSON: {}",
        serde_json::to_string_pretty(&frontend_puzzle).unwrap()
    );

    println!(
        "Puzzle: {}",
        generate_ptn_ninja_link(
            S,
            &puzzle.root_tps,
            &puzzle.defender_start_move,
            &puzzle.player_white,
            &puzzle.player_black,
            Komi::from_half_komi(puzzle.half_komi as i8).unwrap(),
        )
    );

    let moves_with_solution = format!("{} {}", puzzle.defender_start_move, puzzle.solution);

    println!(
        "Solution: {}",
        generate_ptn_ninja_link(
            S,
            &puzzle.root_tps,
            &moves_with_solution,
            "White",
            "Black",
            Komi::default(),
        )
    );
}

/// Checks that the position is a forced win in 2 ply (so a loss for side-to-move),
/// and returns the most challenging move for the defending side, based on some heuristics.
/// If the best defending move only allow one winning move for the attacking side, it returns that move as well
fn find_last_defending_move<const S: usize>(
    position: &mut Position<S>,
) -> Option<(Move<S>, Option<Move<S>>)> {
    let mut defending_moves = vec![];
    position.generate_moves(&mut defending_moves);
    find_last_defending_move_among_moves(position, &defending_moves)
}

fn find_last_defending_move_among_moves<const S: usize>(
    position: &mut Position<S>,
    defending_moves: &[Move<S>],
) -> Option<(Move<S>, Option<Move<S>>)> {
    assert!(defending_moves.iter().all(|mv| position.move_is_legal(*mv)),);
    let results: Vec<_> = defending_moves
        .into_iter()
        .map(|defending_move| {
            let reverse_move = position.do_move(*defending_move);

            let mut moves = vec![];
            position.generate_moves(&mut moves);
            let mut num_winning_moves = 0;
            let mut winning_move = None;
            for mv in moves {
                let reverse_move = position.do_move(mv);
                if position.game_result() == Some(GameResult::win_by(!position.side_to_move())) {
                    winning_move = Some(mv);
                    num_winning_moves += 1;
                }
                position.reverse_move(reverse_move);
            }

            position.reverse_move(reverse_move);
            (defending_move, num_winning_moves, winning_move)
        })
        .collect();
    let (_, lowest_winning_moves, _) = results
        .iter()
        .min_by_key(|(_, num_winning_moves, _)| *num_winning_moves)
        .unwrap()
        .clone();

    if lowest_winning_moves == 0 {
        // At least one defending move leads to non win
        return None;
    }

    let their_reserves_left = match !position.side_to_move() {
        Color::White => position.white_reserves_left() + position.white_caps_left(),
        Color::Black => position.black_reserves_left() + position.black_caps_left(),
    };

    let (best_move, _, winning_response) = results
        .into_iter()
        .filter(|(_, num_winning_moves, _)| *num_winning_moves == lowest_winning_moves)
        .max_by_key(|(mv, _, _)| {
            let mut score = 0;
            // If they're about to win on flats, strongly prefer the highest FCD move
            if their_reserves_left == 1 {
                let fcd = position.fcd_for_move(**mv).max(0);
                score += 12 * fcd;
            }
            if matches!(mv.expand(), ExpMove::Place(Role::Wall, _)) {
                score += 10; // Prefer wall placements
            }
            if position
                .moves()
                .last()
                .unwrap()
                .destination_square()
                .neighbors()
                .any(|neighbor| neighbor == mv.destination_square())
            {
                score += 2; // Prefer moves that are adjacent to the last attacking move
            }
            if position.moves().last().unwrap().origin_square() == mv.destination_square() {
                score += 1; // Prefer moves behind the last attacking move
            }
            score
        })
        .unwrap();

    Some((
        *best_move,
        if lowest_winning_moves == 1 {
            winning_response
        } else {
            None
        },
    ))
}

/// One-off script for extending solutions to 3-ply tinue puzzles, where the solution was erronously only one move long,
/// but any defending move leads to an immediate win
fn extend_tinue_puzzles<const S: usize>(db_path: &str) {
    let puzzles_conn = Connection::open(db_path).unwrap();

    let tinue_puzzles: Vec<FullTinuePuzzle> = puzzles_conn
        .prepare("SELECT * FROM full_tinue_puzzles WHERE end_in_road = 0")
        .unwrap()
        .query_and_then([], from_row::<FullTinuePuzzle>)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    let extendable_puzzles: Vec<FullTinuePuzzle> = tinue_puzzles
        .iter()
        .filter(|puzzle| {
            let solution_length = puzzle.solution.split_whitespace().count();
            puzzle.root_tinue_length == solution_length + 2
        })
        .cloned()
        .collect();

    for puzzle in extendable_puzzles.iter() {
        let mut position: Position<S> = Position::from_fen(&puzzle.root_tps).unwrap();
        position.do_move(Move::from_string(&puzzle.defender_start_move).unwrap());
        let start_tps = position.to_fen();
        for mv_str in puzzle.solution.split_whitespace() {
            let mv = Move::from_string(mv_str).unwrap();
            assert!(position.move_is_legal(mv), "Illegal move: {}", mv);
            position.do_move(mv);
        }
        if position.game_result().is_some() || puzzle.root_tinue_length % 2 == 0 {
            println!(
                "Found unexpected winning position in puzzle #{}",
                puzzle.playtak_game_id
            );
            continue; // Skip puzzles that are already solved
        }

        let last_defending_move = find_last_defending_move(&mut position);

        if let Some((last_move, unique_win)) = last_defending_move {
            let mut new_solution = format!("{} {}", puzzle.solution, last_move);
            if let Some(unique_win) = unique_win {
                write!(new_solution, " {}", unique_win).unwrap();
            }
            let rows_affected = puzzles_conn
            .execute(
                "UPDATE full_tinue_puzzles SET end_in_road = 1, solution = ?1 WHERE root_tps = ?2 AND defender_start_move = ?3",
                [&new_solution, &puzzle.root_tps, &puzzle.defender_start_move],
            )
            .unwrap();
            println!(
                "Updated puzzle, affected {} rows, new solution {}",
                rows_affected, new_solution
            );
        }

        println!(
            "#{}: TPS {}\nsolution: {}, tinue length {}\nBest defending move: {:?}",
            puzzle.playtak_game_id,
            start_tps,
            puzzle.solution,
            puzzle.root_tinue_length,
            last_defending_move.map(|(mv, unique_win)| format!(
                "{}, unique win: {:?}",
                mv,
                unique_win.map(|mv| mv.to_string())
            ))
        );
        println!();
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TinueFollowup<const S: usize> {
    parent_tps: String,
    #[serde(deserialize_with = "deseralize_move")]
    parent_move: Move<S>,
    #[serde(deserialize_with = "deseralize_move")]
    solution: Move<S>,
    tinue_length: usize,
    longest_parent_tinue_length: usize,
    tiltak_0komi_eval: f32,
    tiltak_0komi_second_move_eval: f32,
    tiltak_0komi_pv_length: u32,
    tiltak_0komi_second_pv_length: u32,
    #[serde(deserialize_with = "deseralize_move")]
    tiltak_0komi_move: Move<S>,
    tiltak_2komi_eval: f32,
    tiltak_2komi_second_move_eval: f32,
    tiltak_2komi_pv_length: u32,
    tiltak_2komi_second_pv_length: u32,
    #[serde(deserialize_with = "deseralize_move")]
    tiltak_2komi_move: Move<S>,
}

impl<const S: usize> TinueFollowup<S> {
    fn score_0komi(&self) -> f32 {
        self.tinue_length as f32 - self.longest_parent_tinue_length as f32 // Strongly prioritize the longest tinue defense
        + if self.tiltak_0komi_move == self.solution { // Strong bonus when Tiltak eval isn't conclusively winning, and exceptionally strong if Tiltak also chooses the wrong move
            (1.0 - self.tiltak_0komi_eval) * 6.0
        } else {
            (1.0 - self.tiltak_0komi_eval) * 12.0
        }
        + (0.9 - self.tiltak_0komi_second_move_eval) * 3.0 // Prefer the second move to have lower eval, i.e. the solution is the only strong move in the position
    }

    fn score_2komi(&self) -> f32 {
        self.tinue_length as f32 - self.longest_parent_tinue_length as f32 // Strongly prioritize the longest tinue defense
        + if self.tiltak_2komi_move == self.solution { // Strong bonus when Tiltak eval isn't conclusively winning, and exceptionally strong if Tiltak also chooses the wrong move
            (1.0 - self.tiltak_2komi_eval) * 6.0
        } else {
            (1.0 - self.tiltak_2komi_eval) * 12.0
        }
        + (0.9 - self.tiltak_2komi_second_move_eval) * 3.0 // Prefer the second move to have lower eval, i.e. the solution is the only strong move in the position
    }
}

#[derive(Debug)]
pub enum PuzzleF<const S: usize> {
    UniqueTinue(TinueFollowup<S>),
    NonUniqueTinue,
    UniqueRoadWin(Move<S>, Move<S>),
    NonUniqueRoadWin,
}

pub struct GaeletRoot<const S: usize> {
    playtak_game_id: u32,
    komi: Komi,
    tps: String,
    solution: Move<S>,
    tiltak_eval: f32,
    tiltak_second_move_eval: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImmediateWinRow {
    pub game_id: u64,
    pub tps: String,
    pub tiltak_shallow_alternative_eval: f32,
    pub tiltak_deep_alternative_eval: Option<f32>,
    pub num_winning_moves: usize,
    pub num_winning_origin_squares: usize,
    pub win_type: WinType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum WinType {
    Flat,
    Road,
    Both,
}

/// Returns all positions that are immediate wins in the game
fn immediate_wins<const S: usize>(
    game: &PlaytakGame,
    games_processed: &AtomicU64,
    start_time: &Instant,
) -> Vec<ImmediateWinRow> {
    if game.komi.half_komi() != 0 && game.komi.half_komi() != 4 {
        return vec![];
    }
    let mut wins = vec![];
    let mut position: Position<S> = Position::start_position_with_komi(game.komi);
    for mv in game
        .notation
        .split_whitespace()
        .map(|move_str| Move::from_string(move_str).unwrap())
    {
        assert!(
            position.move_is_legal(mv),
            "{} is illegal in #{} on {}, notation: {}",
            mv,
            game.id,
            position.to_fen(),
            game.notation
        );
        position.do_move(mv);
        if position.game_result().is_some() {
            continue;
        }
        let mut legal_moves = vec![];
        position.generate_moves(&mut legal_moves);

        let mut winning_moves: Vec<(Move<S>, &str)> = vec![];
        for mv in legal_moves {
            let reverse_move = position.do_move(mv);
            if position.game_result() == Some(GameResult::win_by(!position.side_to_move())) {
                winning_moves.push((mv, position.pgn_game_result().unwrap()));
            }
            position.reverse_move(reverse_move);
        }

        if !winning_moves.is_empty() {
            let win_type = if winning_moves
                .iter()
                .any(|(_, result_str)| *result_str != winning_moves[0].1)
            {
                WinType::Both
            } else if winning_moves[0].1.contains('F') {
                WinType::Flat
            } else {
                WinType::Road
            };

            let mut shallow_tree = MonteCarloTree::new(
                position.clone(),
                MctsSetting::default()
                    .arena_size_for_nodes(TILTAK_SHALLOW_NODES)
                    .exclude_moves(winning_moves.iter().map(|(mv, _)| *mv).collect()),
            );

            for _ in 0..TILTAK_SHALLOW_NODES {
                match shallow_tree.select() {
                    Ok(_) => (),
                    Err(err) => {
                        eprintln!("Tiltak search aborted early: {}", err);
                        break;
                    }
                }
            }

            let (_, shallow_eval) = shallow_tree.best_move().unwrap();

            let deep_eval = if shallow_eval < 0.95 {
                let mut deep_tree = MonteCarloTree::new(
                    position.clone(),
                    MctsSetting::default()
                        .arena_size_for_nodes(TILTAK_DEEP_NODES)
                        .exclude_moves(winning_moves.iter().map(|(mv, _)| *mv).collect()),
                );

                for _ in 0..TILTAK_DEEP_NODES {
                    match deep_tree.select() {
                        Ok(_) => (),
                        Err(err) => {
                            eprintln!("Tiltak search aborted early: {}", err);
                            break;
                        }
                    }
                }

                let (_, deep_eval) = deep_tree.best_move().unwrap();
                Some(deep_eval)
            } else {
                None
            };

            wins.push(ImmediateWinRow {
                game_id: game.id,
                tps: position.to_fen(),
                tiltak_shallow_alternative_eval: shallow_eval,
                tiltak_deep_alternative_eval: deep_eval,
                num_winning_moves: winning_moves.len(),
                num_winning_origin_squares: winning_moves
                    .iter()
                    .map(|mv| mv.0.origin_square().into_inner())
                    .collect::<BTreeSet<_>>()
                    .len(),
                win_type,
            });
        }
    }
    let games_processed = games_processed.fetch_add(1, Ordering::Relaxed) + 1;
    if games_processed % 100 == 0 {
        println!(
            "Processed {} games in {:.1}",
            games_processed,
            start_time.elapsed().as_secs_f32()
        );
    }
    wins
}

pub fn find_immediate_wins<const S: usize>(db_path: &str) {
    let puzzles_conn = Connection::open(db_path).unwrap();

    puzzles_conn
        .execute(
            "CREATE TABLE IF NOT EXISTS immediate_wins (
            game_id	INTEGER NOT NULL,
            tps	TEXT PRIMARY KEY,
            tiltak_shallow_alternative_eval REAL NOT NULL,
            tiltak_deep_alternative_eval REAL,
            num_winning_moves INTEGER NOT NULL,
            num_winning_origin_squares INTEGER NOT NULL,
            win_type TEXT NOT NULL,
            FOREIGN KEY(game_id) REFERENCES games(id)
            )",
            [],
        )
        .unwrap();

    // Check all games for unique flat wins with a spread
    let all_games = read_all_games::<S>(&puzzles_conn);

    let start_time = Instant::now();
    let games_processed = Arc::new(AtomicU64::new(0));

    println!("Finding immediate wins from {} games", all_games.len());

    let immediate_wins: Vec<ImmediateWinRow> = all_games
        .par_iter()
        .flat_map(|game| immediate_wins::<S>(game, &games_processed, &start_time))
        .map_init(|| Connection::open(db_path).unwrap(), |puzzles_conn, win| {
            puzzles_conn.execute(
                "INSERT OR IGNORE INTO immediate_wins VALUES (:game_id, :tps, :tiltak_shallow_alternative_eval, :tiltak_deep_alternative_eval, :num_winning_moves, :num_winning_origin_squares, :win_type)",
                to_params_named(&win).unwrap().to_slice().as_slice()
            ).unwrap();
            win
        })
        .collect();

    println!(
        "Found {} immediate wins from {} total games in {:.1}s",
        immediate_wins.len(),
        all_games.len(),
        start_time.elapsed().as_secs_f32()
    );
}

pub fn find_root_gaelets<const S: usize>(db_path: &str) {
    let puzzles_conn = Connection::open(db_path).unwrap();

    let puzzle_roots_0_komi: Vec<GaeletRoot<S>> = puzzles_conn.prepare("SELECT games.id, puzzles.tps, puzzles.solution, puzzles.tiltak_0komi_eval, puzzles.tiltak_0komi_second_move_eval FROM puzzles JOIN games ON puzzles.game_id = games.id
        WHERE games.size = ?1 AND games.komi = 0 AND tinue_length IS NULL AND tinue_avoidance_length IS NULL AND tiltak_0komi_eval > 0.9 AND tiltak_0komi_second_move_eval < 0.6")
    .unwrap()
        .query([S])
        .unwrap()
        .mapped(|row| {
            Ok(GaeletRoot {
                playtak_game_id: row.get::<_, u32>(0).unwrap(),
                komi: Komi::from_half_komi(0).unwrap(),
                tps: row.get(1).unwrap(),
                solution: Move::from_string(&row.get::<_, String>(2).unwrap()).unwrap(),
                tiltak_eval: row.get::<_, f32>(3).unwrap(),
                tiltak_second_move_eval: row.get::<_, f32>(4).unwrap(),
            })
        })
        .map(|row| row.unwrap())
        .collect();

    let puzzle_roots_2_komi: Vec<GaeletRoot<S>> = puzzles_conn.prepare("SELECT games.id, puzzles.tps, puzzles.solution, puzzles.tiltak_2komi_eval, puzzles.tiltak_2komi_second_move_eval FROM puzzles JOIN games ON puzzles.game_id = games.id
        WHERE games.size = ?1 AND games.komi = 4 AND tinue_length IS NULL AND tinue_avoidance_length IS NULL AND tiltak_2komi_eval > 0.9 AND tiltak_2komi_second_move_eval < 0.6")
    .unwrap()
        .query([S])
        .unwrap()
        .mapped(|row| {
            Ok(GaeletRoot {
                playtak_game_id: row.get::<_, u32>(0).unwrap(),
                komi: Komi::from_half_komi(4).unwrap(),
                tps: row.get(1).unwrap(),
                solution: Move::from_string(&row.get::<_, String>(2).unwrap()).unwrap(),
                tiltak_eval: row.get::<_, f32>(3).unwrap(),
                tiltak_second_move_eval: row.get::<_, f32>(4).unwrap(),
            })
        })
        .map(|row| row.unwrap())
        .collect();

    let puzzle_roots: Vec<_> = puzzle_roots_0_komi
        .iter()
        .chain(&puzzle_roots_2_komi)
        .collect();

    let immediate_wins: Vec<_> = puzzle_roots
        .iter()
        .filter(|puzzle| {
            let mut position: Position<S> = Position::from_fen(&puzzle.tps).unwrap();
            position.do_move(puzzle.solution);
            position.game_result().is_some()
        })
        .collect();

    let endgames: Vec<_> = puzzle_roots
        .iter()
        .filter(|puzzle| {
            let position: Position<S> = Position::from_fen(&puzzle.tps).unwrap();
            position.white_reserves_left() <= 3 || position.black_reserves_left() <= 3
        })
        .collect();

    println!(
        "Found {} positions with one best move, {} immediate wins, {} end game positions",
        puzzle_roots.len(),
        immediate_wins.len(),
        endgames.len()
    );
    for endgame in immediate_wins.iter().take(30) {
        println!(
            "#{}: {} komi, {}",
            endgame.playtak_game_id, endgame.komi, endgame.tps
        );
        println!(
            "Solution: {}, Tiltak eval: {:.3}, Tiltak 2nd move eval: {:.3}",
            endgame.solution, endgame.tiltak_eval, endgame.tiltak_second_move_eval
        );
    }

    println!("====");

    for endgame in endgames.iter().take(10) {
        println!(
            "#{}: {} komi, {}",
            endgame.playtak_game_id, endgame.komi, endgame.tps
        );
        println!(
            "Solution: {}, Tiltak eval: {:.3}, Tiltak 2nd move eval: {:.3}",
            endgame.solution, endgame.tiltak_eval, endgame.tiltak_second_move_eval
        );
    }
}

pub struct FullTinuePuzzleOption<const S: usize> {
    playtak_game_id: u32,
    tps: String,
    root_tinue_length: usize,
    solutions: Vec<(Vec<Move<S>>, bool)>,
}

#[derive(Clone, Serialize, Deserialize)]
struct FullTinuePuzzle {
    root_tps: String,
    defender_start_move: String,
    playtak_game_id: usize,
    root_tinue_length: usize,
    solution: String,
    end_in_road: bool,
    first_move_rating: Option<usize>,
    total_rating: Option<usize>,
}

impl FullTinuePuzzle {
    fn from_puzzle_option<const S: usize>(
        puzzle_option: &FullTinuePuzzleOption<S>,
        playtak_game: &PlaytakGame,
        solution_index: usize,
    ) -> Self {
        let mut position: Position<S> = Position::start_position();
        let (root_tps, defender_start_move) = playtak_game
            .notation
            .split_whitespace()
            .find_map(|move_string| {
                let mv = Move::from_string(&move_string).unwrap();
                let last_tps = position.to_fen();
                position.do_move(mv);
                let tps = position.to_fen();
                if tps == puzzle_option.tps {
                    Some((last_tps, mv))
                } else {
                    None
                }
            })
            .unwrap();
        FullTinuePuzzle {
            root_tps,
            defender_start_move: defender_start_move.to_string(),
            playtak_game_id: puzzle_option.playtak_game_id as usize,
            root_tinue_length: puzzle_option.root_tinue_length,
            solution: puzzle_option.solutions[solution_index]
                .0
                .iter()
                .map(|mv| mv.to_string())
                .collect::<Vec<String>>()
                .join(" "),
            end_in_road: puzzle_option.solutions[solution_index].1,
            first_move_rating: None,
            total_rating: None,
        }
    }
}

impl<const S: usize> FullTinuePuzzleOption<S> {
    /// Returns a single solution if the puzzle has only one, or if it has exactly one that goes to road
    fn single_solution(&self) -> Option<(Vec<Move<S>>, bool)> {
        if self.solutions.len() == 1 {
            Some(self.solutions[0].clone())
        } else if self
            .solutions
            .iter()
            .filter(|(_, goes_to_road)| *goes_to_road)
            .count()
            == 1
        {
            self.solutions
                .iter()
                .cloned()
                .find(|(_, goes_to_road)| *goes_to_road)
                .clone()
        } else {
            None
        }
    }
}

fn generate_ptn_ninja_link(
    size: usize,
    root_tps: &str,
    moves: &str,
    player_white: &str,
    player_black: &str,
    komi: Komi,
) -> String {
    let ptn = format!(
        "[Size \"{}\"]
            [TPS \"{}\"]
            [Player1 \"{}\"]
            [Player2 \"{}\"]
            [Komi \"{}\"]
                {}
            ",
        size, root_tps, player_white, player_black, komi, moves,
    );

    format!(
        "Url: https://ptn.ninja/{}&ply=1",
        lz_str::compress_to_encoded_uri_component(&ptn)
    )
}

fn insert_full_puzzles<const S: usize>(
    db_conn: &mut Connection,
    puzzles: &[FullTinuePuzzleOption<S>],
) {
    db_conn
        .execute(
            "CREATE TABLE IF NOT EXISTS full_tinue_puzzles (
                root_tps TEXT NOT NULL,
                defender_start_move TEXT NOT NULL,
                playtak_game_id INTEGER NOT NULL,
                root_tinue_length INTEGER NOT NULL,
                solution TEXT NOT NULL,
                end_in_road INTEGER NOT NULL,
                first_move_rating INTEGER,
                total_rating INTEGER,
                PRIMARY KEY (root_tps, defender_start_move)
            )",
            [],
        )
        .unwrap();

    let all_playtak_games = read_all_games::<S>(db_conn);
    let playtak_games_map = all_playtak_games
        .iter()
        .map(|game| (game.id, game.clone()))
        .collect::<HashMap<u64, PlaytakGame>>();

    let num_puzzles = puzzles.len();
    let num_single_solution_to_road = puzzles
        .iter()
        .filter(|puzzle| matches!(puzzle.single_solution(), Some((_, true))))
        .count();
    let num_to_road = puzzles
        .iter()
        .filter(|puzzle| {
            puzzle
                .solutions
                .iter()
                .any(|(_, goes_to_road)| *goes_to_road)
        })
        .count();
    println!(
        "Got {} full puzzles, {} with single solution to road, {} with at least one solution that goes to road",
        num_puzzles, num_single_solution_to_road, num_to_road
    );

    for (i, puzzle) in puzzles.iter().enumerate() {
        let playtak_game = &playtak_games_map[&(puzzle.playtak_game_id as u64)];
        match puzzle.single_solution() {
            // A single solution that goes to road does not need manual review
            Some((_, true)) => {
                println!("Automatically added puzzle");
                let full_puzzle = FullTinuePuzzle::from_puzzle_option(puzzle, playtak_game, 0);

                let rows_affected = db_conn.execute("INSERT OR IGNORE INTO full_tinue_puzzles VALUES (:root_tps, :defender_start_move, :playtak_game_id, :root_tinue_length, :solution, :end_in_road, :first_move_rating, :total_rating)", to_params_named(&full_puzzle).unwrap().to_slice().as_slice()).unwrap();
                if rows_affected != 1 {
                    println!("Warning: {} rows affected by insert", rows_affected);
                }
            }
            _ => {
                let tinue = puzzle;
                println!("==========================");
                println!(
                    "Puzzle {}/{} TPS: {}, tinue length {}",
                    i, num_puzzles, tinue.tps, tinue.root_tinue_length
                );
                let ptn = format!(
                    "[Size \"{}\"]
            [TPS \"{}\"]
            [Player1 \"{}\"]
            [Player2 \"{}\"]
            [Komi \"{}\"]
                {}
            ",
                    S,
                    tinue.tps,
                    playtak_game.player_white,
                    playtak_game.player_black,
                    playtak_game.komi,
                    tinue.solutions[0]
                        .0
                        .iter()
                        .map(|mv| mv.to_string())
                        .collect::<Vec<String>>()
                        .join(" "),
                );

                println!(
                    "Url: https://ptn.ninja/{}&ply=0",
                    lz_str::compress_to_encoded_uri_component(&ptn)
                );
                let mut line_strings = vec![];
                for (line, goes_to_road) in tinue.solutions.iter() {
                    let mut line_string = line
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" ");
                    if line.len() % 2 == 0 {
                        // If the line's length is even, the last move is any move that wins
                        line_string.push_str(" *");
                    }
                    print!("{}", line_string);
                    line_strings.push(line_string);
                    if *goes_to_road {
                        println!(" (goes to road win)");
                    } else {
                        println!();
                    }
                }

                println!("Copy-paste the solution to add it to the database, type 'skip' to skip");
                loop {
                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();
                    let input = input.trim();
                    if input == "skip" {
                        println!("Skipping puzzle");
                        break;
                    }
                    if input.is_empty() {
                        println!("Empty input, try again");
                        continue;
                    }
                    let solution_index = line_strings.iter().position(|s| s == input);
                    let Some(solution_index) = solution_index else {
                        println!("Solution not found in the list, try again");
                        continue;
                    };

                    let full_puzzle =
                        FullTinuePuzzle::from_puzzle_option(puzzle, playtak_game, solution_index);

                    let rows_affected = db_conn.execute("INSERT OR IGNORE INTO full_tinue_puzzles VALUES (:root_tps, :defender_start_move, :playtak_game_id, :root_tinue_length, :solution, :end_in_road, :first_move_rating, :total_rating)", to_params_named(&full_puzzle).unwrap().to_slice().as_slice()).unwrap();
                    if rows_affected != 1 {
                        println!("Warning: {} rows affected by insert", rows_affected);
                    }
                    break;
                }
            }
        }
    }
}

fn find_full_puzzles<const S: usize>(db_path: &str) {
    let mut puzzles_conn = Connection::open(db_path).unwrap();

    let puzzle_roots: Vec<PuzzleRoot<S>> = puzzles_conn.prepare("SELECT puzzles.tps, puzzles.solution, puzzles.tinue_length, games.id FROM puzzles JOIN games ON puzzles.game_id = games.id
        WHERE games.size = ?1 AND puzzles.tinue_length NOT NULL AND (tiltak_0komi_second_move_eval < 0.7 OR tiltak_2komi_second_move_eval < 0.7)")
    .unwrap()
        .query([S])
        .unwrap()
        .mapped(|row| {
            Ok(PuzzleRoot {
                playtak_game_id: row.get::<_, u32>(3).unwrap(),
                tps: row.get(0).unwrap(),
                solution: Move::from_string(&row.get::<_, String>(1).unwrap()).unwrap(),
                tinue_length: row.get(2).unwrap(),
            })
        })
        .map(|row| row.unwrap())
        .collect();

    println!("Found {} puzzles roots", puzzle_roots.len());
    let full_tinues: Vec<FullTinuePuzzleOption<S>> =
        extract_possible_full_tinues(&mut puzzles_conn, &puzzle_roots);

    insert_full_puzzles(&mut puzzles_conn, &full_tinues);
}

pub fn extract_possible_full_tinues<const S: usize>(
    puzzles_conn: &mut Connection,
    puzzle_roots: &[PuzzleRoot<S>],
) -> Vec<FullTinuePuzzleOption<S>> {
    let mut full_tinues: Vec<FullTinuePuzzleOption<S>> = vec![];

    for puzzle_root in puzzle_roots.iter() {
        let mut position = Position::from_fen(&puzzle_root.tps).unwrap();

        let mut moves = vec![puzzle_root.solution];
        let mut possible_lines = vec![];

        position.do_move(puzzle_root.solution);

        find_followup_recursive(puzzles_conn, &mut position, &mut moves, &mut possible_lines);

        let tinue = FullTinuePuzzleOption {
            playtak_game_id: puzzle_root.playtak_game_id,
            tps: puzzle_root.tps.clone(),
            root_tinue_length: puzzle_root.tinue_length,
            solutions: possible_lines.clone(),
        };
        full_tinues.push(tinue);
    }

    full_tinues
}

fn find_followup_recursive<const S: usize>(
    puzzles_conn: &mut Connection,
    position: &mut Position<S>,
    moves: &mut Vec<Move<S>>,
    possible_lines: &mut Vec<(Vec<Move<S>>, bool)>,
) {
    let road_win_followup = read_road_win_followup::<S>(puzzles_conn, &position.to_fen());
    if let Some(road_win_move) = road_win_followup {
        let reverse_move = position.do_move(road_win_move);
        moves.push(road_win_move);

        let mut unique_winning_move: Option<Move<S>> = None;
        let mut legal_moves = vec![];
        position.generate_moves(&mut legal_moves);
        for legal_move in legal_moves {
            let reverse_move = position.do_move(legal_move);
            if position.game_result() == Some(GameResult::win_by(!position.side_to_move())) {
                if unique_winning_move.is_some() {
                    // Winning move wasn't unique
                    unique_winning_move = None;
                    position.reverse_move(reverse_move);
                    break;
                }
                unique_winning_move = Some(legal_move);
            }
            position.reverse_move(reverse_move);
        }
        if let Some(unique_winning_move) = unique_winning_move {
            moves.push(unique_winning_move);
        }

        possible_lines.push((moves.clone(), true));
        if unique_winning_move.is_some() {
            moves.pop();
        }
        position.reverse_move(reverse_move);
        moves.pop();
        return;
    }
    let followups = read_followup::<S>(puzzles_conn, &position.to_fen());

    if followups.is_empty() {
        possible_lines.push((moves.clone(), false));
        return;
    }
    for followup in followups {
        let reverse_move = position.do_move(followup.parent_move);
        let reverse_move2 = position.do_move(followup.solution);
        moves.push(followup.parent_move);
        moves.push(followup.solution);

        find_followup_recursive(puzzles_conn, position, moves, possible_lines);

        moves.pop();
        moves.pop();

        position.reverse_move(reverse_move2);
        position.reverse_move(reverse_move);
    }
}

fn read_followup<const S: usize>(puzzles_db: &mut Connection, tps: &str) -> Vec<TinueFollowup<S>> {
    let mut stmt = puzzles_db
        .prepare("SELECT * FROM tinue_followups WHERE parent_tps = ?1")
        .unwrap();
    let result: Vec<TinueFollowup<S>> = from_rows::<TinueFollowup<S>>(stmt.query([tps]).unwrap())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    result
}

pub fn find_followup<const S: usize>(
    puzzle_root: &PuzzleRoot<S>,
    stats: &Stats,
) -> Vec<PuzzleF<S>> {
    let mut position: Position<S> =
        Position::from_fen_with_komi(&puzzle_root.tps, Komi::default()).unwrap();
    assert!(position.move_is_legal(puzzle_root.solution));
    position.do_move(puzzle_root.solution);

    let parent_tps = position.to_fen();

    let mut legal_moves = vec![];
    position.generate_moves(&mut legal_moves);

    let mut longest_tinue_length = 0;

    let mut followups: Vec<PuzzleF<S>> = legal_moves
        .iter()
        .filter_map(|mv| {
            let reverse_move = position.do_move(*mv);
            if position.game_result().is_some() {
                position.reverse_move(reverse_move);
                return None;
            }

            let mut komi_position = position.clone();
            komi_position.set_komi(Komi::from_half_komi(4).unwrap());

            // Check if this response gives us one or more immediate winning moves
            let mut immediate_winning_move = None;
            let mut child_moves = vec![];
            position.generate_moves(&mut child_moves);
            for child_move in child_moves {
                let reverse_child_move = position.do_move(child_move);
                if position.game_result() == Some(GameResult::win_by(!position.side_to_move())) {
                    if immediate_winning_move.is_some() {
                        position.reverse_move(reverse_child_move);
                        position.reverse_move(reverse_move);
                        return Some(PuzzleF::NonUniqueRoadWin);
                    }
                    immediate_winning_move = Some(child_move);
                }
                position.reverse_move(reverse_child_move);
            }

            if let Some(win) = immediate_winning_move {
                position.reverse_move(reverse_move);
                return Some(PuzzleF::UniqueRoadWin(*mv, win));
            }

            let topaz_result = topaz_search::<S>(&position.to_fen(), stats);
            let result = if let TopazResult::Tinue(tinue) = topaz_result {
                longest_tinue_length = longest_tinue_length.max(tinue.len());

                let tiltak_start_time = Instant::now();
                let tiltak_0komi_analysis_deep = tiltak_search(position.clone(), TILTAK_DEEP_NODES);

                let tiltak_2komi_analysis_deep =
                    tiltak_search(komi_position.clone(), TILTAK_DEEP_NODES);

                stats
                    .tiltak_non_tinue_long
                    .record(tiltak_start_time.elapsed());

                Some(PuzzleF::UniqueTinue(TinueFollowup {
                    parent_tps: parent_tps.clone(),
                    parent_move: *mv,
                    solution: *tinue.first().unwrap(),
                    tinue_length: tinue.len(),
                    longest_parent_tinue_length: 0,
                    tiltak_0komi_eval: tiltak_0komi_analysis_deep.score_first,
                    tiltak_0komi_second_move_eval: tiltak_0komi_analysis_deep.score_second,
                    tiltak_0komi_pv_length: tiltak_0komi_analysis_deep.pv_first.len() as u32,
                    tiltak_0komi_second_pv_length: tiltak_0komi_analysis_deep.pv_second.len()
                        as u32,
                    tiltak_0komi_move: tiltak_0komi_analysis_deep.pv_first[0],
                    tiltak_2komi_eval: tiltak_2komi_analysis_deep.score_first,
                    tiltak_2komi_second_move_eval: tiltak_2komi_analysis_deep.score_second,
                    tiltak_2komi_pv_length: tiltak_2komi_analysis_deep.pv_first.len() as u32,
                    tiltak_2komi_second_pv_length: tiltak_2komi_analysis_deep.pv_second.len()
                        as u32,
                    tiltak_2komi_move: tiltak_2komi_analysis_deep.pv_first[0],
                }))
            } else if let TopazResult::NonUniqueTinue(tinue) = topaz_result {
                longest_tinue_length = longest_tinue_length.max(tinue.len());
                Some(PuzzleF::NonUniqueTinue)
            } else {
                None
            };
            position.reverse_move(reverse_move);
            result
        })
        .collect();

    for followup in followups.iter_mut() {
        if let PuzzleF::UniqueTinue(tinue) = followup {
            tinue.longest_parent_tinue_length = longest_tinue_length;
        }
    }
    followups
}

fn store_road_win_followup(conn: &mut Connection, parent_tps: &str, parent_move_string: &str) {
    conn.execute(
        "INSERT OR IGNORE INTO road_win_followups (parent_tps, parent_move) values (?1, ?2)",
        rusqlite::params![parent_tps, parent_move_string],
    )
    .unwrap();
}

fn read_road_win_followup<const S: usize>(
    conn: &mut Connection,
    parent_tps: &str,
) -> Option<Move<S>> {
    let mut stmt = conn
        .prepare("SELECT parent_move FROM road_win_followups WHERE parent_tps = ?1")
        .unwrap();
    let result: Option<String> = stmt
        .query_row([parent_tps], |row| Ok(row.get::<_, String>(0).unwrap()))
        .ok();
    result.and_then(|s| Move::from_string(&s).ok())
}

fn find_followups<const S: usize>(db_path: &str) {
    let stats = Arc::new(Stats::default());

    let puzzles_conn = Connection::open(db_path).unwrap();
    let mut stmt = puzzles_conn.prepare("SELECT puzzles.tps, puzzles.solution, puzzles.tinue_length, games.id FROM puzzles JOIN games ON puzzles.game_id = games.id
        WHERE games.size = ?1 AND puzzles.tinue_length NOT NULL AND (tiltak_0komi_second_move_eval < 0.7 OR tiltak_2komi_second_move_eval < 0.7) AND puzzles.followups_analyzed = 0")
    .unwrap();
    let puzzle_roots: Vec<PuzzleRoot<S>> = stmt
        .query([S])
        .unwrap()
        .mapped(|row| {
            Ok(PuzzleRoot {
                playtak_game_id: row.get::<_, u32>(3).unwrap(),
                tps: row.get(0).unwrap(),
                solution: Move::from_string(&row.get::<_, String>(1).unwrap()).unwrap(),
                tinue_length: row.get(2).unwrap(),
            })
        })
        .map(|row| row.unwrap())
        .collect();
    println!(
        "Found {} puzzles roots, like {}",
        puzzle_roots.len(),
        puzzle_roots
            .first()
            .map(ToString::to_string)
            .unwrap_or_default()
    );

    puzzles_conn
        .execute(
            "CREATE TABLE IF NOT EXISTS tinue_followups(
            parent_tps TEXT NOT NULL,
            parent_move TEXT NOT NULL,
            solution TEXT NOT NULL,
            tinue_length INTEGER NOT NULL,
            longest_parent_tinue_length INTEGER NOT NULL,

            tiltak_0komi_eval REAL NOT NULL,
            tiltak_2komi_eval REAL NOT NULL,
            tiltak_0komi_second_move_eval REAL NOT NULL,
            tiltak_2komi_second_move_eval REAL NOT NULL,
            tiltak_0komi_move TEXT NOT NULL,
            tiltak_2komi_move TEXT NOT NULL,
            tiltak_0komi_pv_length INTEGER NOT NULL,
            tiltak_2komi_pv_length INTEGER NOT NULL,
            tiltak_0komi_second_pv_length INTEGER NOT NULL,
            tiltak_2komi_second_pv_length INTEGER NOT NULL,

            followups_analyzed INT DEFAULT 0,

            PRIMARY KEY (parent_tps, parent_move)
        )",
            [],
        )
        .unwrap();

    puzzles_conn
        .execute(
            "CREATE TABLE IF NOT EXISTS road_win_followups(
                parent_tps TEXT NOT NULL,
                parent_move TEXT NOT NULL, 

                PRIMARY KEY (parent_tps, parent_move)
            )",
            [],
        )
        .unwrap();

    fn store_tinue_followup<const S: usize>(conn: &mut Connection, followup: &TinueFollowup<S>) {
        conn.execute(
            "
        INSERT OR IGNORE INTO tinue_followups (
            parent_tps,
            parent_move,
            solution,
            tinue_length,
            longest_parent_tinue_length,
            tiltak_0komi_eval,
            tiltak_2komi_eval,
            tiltak_0komi_second_move_eval,
            tiltak_2komi_second_move_eval,
            tiltak_0komi_move,
            tiltak_2komi_move,
            tiltak_0komi_pv_length,
            tiltak_2komi_pv_length,
            tiltak_0komi_second_pv_length,
            tiltak_2komi_second_pv_length,
            followups_analyzed)
            values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 0)",
            rusqlite::params![
                followup.parent_tps,
                followup.parent_move.to_string(),
                followup.solution.to_string(),
                followup.tinue_length,
                followup.longest_parent_tinue_length,
                followup.tiltak_0komi_eval,
                followup.tiltak_2komi_eval,
                followup.tiltak_0komi_second_move_eval,
                followup.tiltak_2komi_second_move_eval,
                followup.tiltak_0komi_move.to_string(),
                followup.tiltak_2komi_move.to_string(),
                followup.tiltak_0komi_pv_length,
                followup.tiltak_2komi_pv_length,
                followup.tiltak_0komi_second_pv_length,
                followup.tiltak_2komi_second_pv_length,
            ],
        )
        .unwrap();
    }

    let start_time = Instant::now();
    let num_root_puzzles = puzzle_roots.len();

    puzzle_roots.par_iter().for_each_init(
        || Connection::open(db_path).unwrap(),
        |conn, puzzle_root| {
            let followups = find_followup(puzzle_root, &stats);

            let mut tinue_followups: Vec<TinueFollowup<S>> = followups
                .iter()
                .filter_map(|followup| match followup {
                    PuzzleF::UniqueTinue(followup_tinue) => Some(followup_tinue.clone()),
                    _ => None,
                })
                .collect();

            tinue_followups.sort_by(|followup1, followup2| {
                followup1
                    .score_0komi()
                    .partial_cmp(&followup2.score_0komi())
                    .unwrap()
                    .reverse()
            });

            let n = NUM_GAMES_PROCESSED.fetch_add(1, Ordering::AcqRel);
            let num_unique_tinues = tinue_followups.len();
            let num_non_unique_tinues = followups
                .iter()
                .filter(|f| matches!(f, PuzzleF::NonUniqueTinue))
                .count();
            let num_unique_road_wins = followups
                .iter()
                .filter(|f| matches!(f, PuzzleF::UniqueRoadWin(_, _)))
                .count();

            println!(
                "{}/{} puzzles processed in {:.1}s, ETA {:.1}s, results for {}:",
                n,
                num_root_puzzles,
                start_time.elapsed().as_secs_f32(),
                (start_time.elapsed().as_secs_f32() / n as f32)
                    * (num_root_puzzles as f32 - n as f32),
                puzzle_root
            );
            println!(
                "Got {} unique tinues, {} num_non_unique_tinues, {:?} longest tinue length",
                num_unique_tinues,
                num_non_unique_tinues,
                tinue_followups
                    .first()
                    .map(|tinue| tinue.longest_parent_tinue_length),
            );

            if !tinue_followups.is_empty() {
                println!("Followups with unique tinue: ");
                for followup in tinue_followups.iter() {
                    println!(
                        "{}: solution {}, score {:.3}, length {}, eval {:.3}, second eval {:.3}",
                        followup.parent_move,
                        followup.solution,
                        followup.score_0komi(),
                        followup.tinue_length,
                        followup.tiltak_0komi_eval,
                        followup.tiltak_0komi_second_move_eval
                    );
                    if followup.score_0komi() > 0.0 || followup.score_2komi() > 0.0 {
                        store_tinue_followup(conn, followup);
                    }
                }
            } else if num_non_unique_tinues > 0 {
                println!(
                    "Got no unique tinues, {} non-unique tinues",
                    num_non_unique_tinues
                );
            } else {
                println!(
                    "Got 0 non unique tinues, {} followups with unique road wins",
                    num_unique_road_wins,
                );
                // If all defenses allow us to make a road immediately, choose Tiltak's top defense move, since they're usually sensible
                let mut position: Position<S> =
                    Position::from_fen_with_komi(&puzzle_root.tps, Komi::default()).unwrap();
                position.do_move(puzzle_root.solution);
                let result = tiltak_search(position.clone(), TILTAK_SHALLOW_NODES);
                store_road_win_followup(conn, &position.to_fen(), &result.pv_first[0].to_string());
            }
            println!();

            set_followup_analyzed(conn, &puzzle_root.tps);
        },
    );
}

/// Creates a games table in the puzzles database, and copies all relevant games form the playtak database
/// The database are separate, so that it's easy to download a new copy of the Playtak db in the future
fn import_playtak_db(playtak_db: &mut Connection, puzzles_db: &mut Connection, size: usize) {
    let games = read_non_bot_games(playtak_db).unwrap();

    let mut relevant_games: Vec<PlaytakGame> = games
        .into_iter()
        .filter(|game| {
            game.size == size
                && game.notation.split_whitespace().count() > 4
                && !game.is_bot_game()
                && game.game_is_legal()
                && game.has_standard_piece_count()
        })
        .collect();

    relevant_games.shuffle(&mut rand::rng());

    puzzles_db
        .execute(
            "CREATE TABLE IF NOT EXISTS games(
            id INT PRIMARY KEY,
            date INT,
            size INT,
            player_white VARCHAR(20),
            player_black VARCHAR(20),
            notation TEXT,
            result VARCAR(10),
            timertime INT NOT NULL,
            timerinc INT NOT NULL,
            rating_white INT NOT NULL,
            rating_black INT NOT NULL,
            unrated INT NOT NULL,
            tournament INT NOT NULL,
            komi INT NOT NULL,
            has_been_analyzed INT DEFAULT 0
        )",
            [],
        )
        .unwrap();

    let mut stmt = puzzles_db.prepare("INSERT OR IGNORE INTO games (
            id, date, size, player_white, player_black, notation, result, timertime, timerinc, rating_white, rating_black, unrated, tournament, komi, has_been_analyzed)
            values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 0)").unwrap();

    println!(
        "Copying up to {} games from the input database. This may take a few minutes.",
        relevant_games.len()
    );
    for game in relevant_games.iter() {
        stmt.execute(rusqlite::params![
            game.id,
            game.date_time,
            game.size,
            game.player_white,
            game.player_black,
            game.notation,
            game.result_string,
            game.game_time.as_secs(),
            game.increment.as_secs(),
            game.rating_white.unwrap_or_default(),
            game.rating_black.unwrap_or_default(),
            game.is_rated,
            game.is_tournament,
            game.komi.half_komi()
        ])
        .unwrap();
    }
}

#[allow(unused)]
fn main_sized<const S: usize>(playtak_db_name: &Option<String>, db_path: &str) {
    let start_time = Instant::now();
    let stats = Arc::new(Stats::default());
    let games_processed = Arc::new(AtomicU64::new(0));

    let mut puzzles_pool = Connection::open(db_path).unwrap();

    if let Some(playtak_db_name) = playtak_db_name {
        let mut db_conn = Connection::open(playtak_db_name).unwrap();
        import_playtak_db(&mut db_conn, &mut puzzles_pool, S);
    } else {
        println!("No Playtak database provided, only previously imported games will be analyzed.");
    }

    puzzles_pool
        .execute(
            "CREATE TABLE IF NOT EXISTS puzzles (
            game_id	INTEGER NOT NULL,
            tps	TEXT PRIMARY KEY,
            solution TEXT NOT NULL,
            tiltak_0komi_eval REAL NOT NULL,
            tiltak_2komi_eval REAL NOT NULL,
            tiltak_0komi_second_move_eval REAL NOT NULL,
            tiltak_2komi_second_move_eval REAL NOT NULL,
            tinue_length INTEGER,
            tinue_avoidance_length INTEGER,
            tiltak_0komi_pv_length INTEGER NOT NULL,
            tiltak_2komi_pv_length INTEGER NOT NULL,
            tiltak_0komi_second_pv_length INTEGER NOT NULL,
            tiltak_2komi_second_pv_length INTEGER NOT NULL,
            followups_analyzed INT DEFAULT 0,
            gaelet_followups_analyzed INT NOT NULL DEFAULT 0,
            FOREIGN KEY(game_id) REFERENCES games(id)
            )",
            [],
        )
        .unwrap();

    puzzles_pool
        .execute(
            "CREATE TABLE IF NOT EXISTS failed_puzzles (
            game_id	INTEGER NOT NULL,
            tps	TEXT PRIMARY KEY,
            failure_kind TEXT,
            FOREIGN KEY(game_id) REFERENCES games(id)
            )",
            [],
        )
        .unwrap();

    puzzles_pool
        .execute(
            "CREATE TABLE IF NOT EXISTS topaz_missed_tinues (
            game_id	INTEGER NOT NULL,
            tps	TEXT PRIMARY KEY,
            FOREIGN KEY(game_id) REFERENCES games(id)
            )",
            [],
        )
        .unwrap();

    let relevant_games = read_processed_games::<S>(&puzzles_pool);

    relevant_games.par_iter().for_each_init(
        || Connection::open(db_path).unwrap(),
        |puzzle_conn, game| {
            for possible_puzzle in generate_possible_puzzle::<S>(&stats, game) {
                if let Some(TopazAvoidanceResult::NoDefense) = possible_puzzle.topaz_tinue_avoidance
                {
                    if !possible_puzzle.last_move_was_tinue {
                        println!(
                            "Found tinue not identified by Topaz, {} in game #{}",
                            possible_puzzle.previous_tps, game.id
                        );
                        store_topaz_missed_tinues(
                            puzzle_conn,
                            game.id as u32,
                            &possible_puzzle.previous_tps,
                        );
                    }
                }
                if let TopazResult::AbortedFirst = possible_puzzle.topaz_tinue {
                    store_failed_puzzle(
                        puzzle_conn,
                        game.id as u32,
                        &possible_puzzle.tps,
                        "Topaz timed out on primary move",
                    )
                } else if let TopazResult::AbortedSecond(_) = possible_puzzle.topaz_tinue {
                    store_failed_puzzle(
                        puzzle_conn,
                        game.id as u32,
                        &possible_puzzle.tps,
                        "Topaz timed out on secondary move",
                    )
                } else if let Some(TopazAvoidanceResult::Aborted) =
                    possible_puzzle.topaz_tinue_avoidance
                {
                    store_failed_puzzle(
                        puzzle_conn,
                        game.id as u32,
                        &possible_puzzle.tps,
                        "Topaz timed out on tinue avoidance",
                    )
                } else if let Some(puzzle) = possible_puzzle.make_real_puzzle() {
                    store_puzzle(puzzle_conn, puzzle);
                }
            }

            set_game_analyzed(puzzle_conn, game.id);

            if games_processed.fetch_add(1, Ordering::SeqCst) % 200 == 0 {
                println!(
                    "Checked {}/{} games in {}s",
                    games_processed.load(Ordering::SeqCst),
                    relevant_games.len(),
                    start_time.elapsed().as_secs()
                );
                print!("Topaz tinue first move: ");
                stats.topaz_tinue_first.print_full();
                print!("Topaz tinue second move: ");
                stats.topaz_tinue_second.print_full();
                print!("Topaz tinue avoidance: ");
                stats.topaz_tinue_avoidance.print_full();
                print!("Tiltak non tinue (short): ");
                stats.tiltak_non_tinue_short.print_short();
                print!("Tiltak non tinue (long): ");
                stats.tiltak_non_tinue_long.print_short();
                print!("Tiltak tinue: ");
                stats.tiltak_tinue.print_short();
                println!();
            }
        },
    );

    println!("Analysis complete");
}

fn set_followup_analyzed(conn: &mut Connection, tps: &str) {
    while let Err(err) = conn.execute(
        "UPDATE puzzles SET followups_analyzed = 1 WHERE tps = ?1",
        [tps],
    ) {
        println!(
            "Failed to update puzzle \"{}\" into DB. Retrying in 1s: {}",
            tps, err
        );
        thread::sleep(Duration::from_secs(1));
    }
}

fn set_game_analyzed(conn: &mut Connection, game_id: u64) {
    while let Err(err) = conn.execute(
        "UPDATE games SET has_been_analyzed = 1 WHERE id = ?1",
        [game_id as u32],
    ) {
        println!(
            "Failed to update game #{} into DB. Retrying in 1s: {}",
            game_id, err
        );
        thread::sleep(Duration::from_secs(1));
    }
}

fn store_failed_puzzle(conn: &mut Connection, game_id: u32, tps: &str, failure_kind: &str) {
    conn.execute(
        "INSERT OR IGNORE INTO failed_puzzles (game_id, tps, failure_kind) values (?1, ?2, ?3)",
        rusqlite::params![game_id, tps, failure_kind],
    )
    .unwrap();
}

fn store_topaz_missed_tinues(conn: &mut Connection, game_id: u32, tps: &str) {
    conn.execute(
        "INSERT OR IGNORE INTO topaz_missed_tinues (game_id, tps) values (?1, ?2)",
        rusqlite::params![game_id, tps],
    )
    .unwrap();
}

fn store_puzzle(puzzles_pool: &mut Connection, puzzle: Puzzle) {
    while let Err(rusqlite::Error::SqliteFailure(err, _)) = puzzles_pool.execute("INSERT OR IGNORE INTO puzzles VALUES (:game_id, :tps, :solution, :tiltak_0komi_eval, :tiltak_2komi_eval, :tiltak_0komi_second_move_eval, :tiltak_2komi_second_move_eval, :tinue_length, :tinue_avoidance_length, :tiltak_0komi_pv_length, :tiltak_2komi_pv_length, :tiltak_0komi_second_pv_length, :tiltak_2komi_second_pv_length, 0, 0)", to_params_named(&puzzle).unwrap().to_slice().as_slice()) {
        println!(
            "Failed to insert \"{}\" from game ${} into DB. Retrying in 1s: {}",
            puzzle.tps, puzzle.game_id, err
        );
        thread::sleep(Duration::from_secs(1));
    }
}

fn read_processed_games<const S: usize>(conn: &Connection) -> Vec<PlaytakGame> {
    let mut stmt = conn.prepare("SELECT id, date, size, player_white, player_black, notation, result, timertime, timerinc, rating_white, rating_black, unrated, tournament, komi FROM games
        WHERE has_been_analyzed = 0 AND size = ?1")
    .unwrap();
    let rows = stmt.query([S]).unwrap().mapped(|row| {
        Ok(GameRow {
            id: row.get(0).unwrap(),
            date: row.get(1).unwrap(),
            size: row.get(2).unwrap(),
            player_white: row.get(3).unwrap(),
            player_black: row.get(4).unwrap(),
            notation: row.get(5).unwrap(),
            result: row.get(6).unwrap(),
            timertime: Duration::from_secs(row.get(7).unwrap()),
            timerinc: Duration::from_secs(row.get(8).unwrap()),
            rating_white: row.get(9).unwrap(),
            rating_black: row.get(10).unwrap(),
            unrated: row.get(11).unwrap(),
            tournament: row.get(12).unwrap(),
            komi: row.get(13).unwrap(),
            pieces: position::starting_stones(row.get(2).unwrap()) as i64,
            capstones: position::starting_capstones(row.get(2).unwrap()) as i64,
        })
    });

    rows.map(|row| {
        let row = row.unwrap();
        PlaytakGame {
            id: row.id as u64,
            date_time: row.date,
            size: row.size as usize,
            player_white: row.player_white,
            player_black: row.player_black,
            notation: row.notation,
            result_string: row.result,
            game_time: row.timertime,
            increment: row.timerinc,
            rating_white: if row.rating_white == 0 {
                None
            } else {
                Some(row.rating_white)
            },
            rating_black: if row.rating_black == 0 {
                None
            } else {
                Some(row.rating_black)
            },
            is_rated: !row.unrated,
            is_tournament: row.tournament,
            komi: Komi::from_half_komi(row.komi as i8).unwrap(),
            flats: row.pieces,
            caps: row.capstones,
        }
    })
    .collect()
}

fn read_all_games<const S: usize>(conn: &Connection) -> Vec<PlaytakGame> {
    let mut stmt = conn.prepare("SELECT id, date, size, player_white, player_black, notation, result, timertime, timerinc, rating_white, rating_black, unrated, tournament, komi FROM games
        WHERE size = ?1")
    .unwrap();
    let rows = stmt.query([S]).unwrap().mapped(|row| {
        Ok(GameRow {
            id: row.get(0).unwrap(),
            date: row.get(1).unwrap(),
            size: row.get(2).unwrap(),
            player_white: row.get(3).unwrap(),
            player_black: row.get(4).unwrap(),
            notation: row.get(5).unwrap(),
            result: row.get(6).unwrap(),
            timertime: Duration::from_secs(row.get(7).unwrap()),
            timerinc: Duration::from_secs(row.get(8).unwrap()),
            rating_white: row.get(9).unwrap(),
            rating_black: row.get(10).unwrap(),
            unrated: row.get(11).unwrap(),
            tournament: row.get(12).unwrap(),
            komi: row.get(13).unwrap(),
            pieces: position::starting_stones(row.get(2).unwrap()) as i64,
            capstones: position::starting_capstones(row.get(2).unwrap()) as i64,
        })
    });

    rows.map(|row| {
        let row = row.unwrap();
        PlaytakGame {
            id: row.id as u64,
            date_time: row.date,
            size: row.size as usize,
            player_white: row.player_white,
            player_black: row.player_black,
            notation: row.notation,
            result_string: row.result,
            game_time: row.timertime,
            increment: row.timerinc,
            rating_white: if row.rating_white == 0 {
                None
            } else {
                Some(row.rating_white)
            },
            rating_black: if row.rating_black == 0 {
                None
            } else {
                Some(row.rating_black)
            },
            is_rated: !row.unrated,
            is_tournament: row.tournament,
            komi: Komi::from_half_komi(row.komi as i8).unwrap(),
            flats: row.pieces,
            caps: row.capstones,
        }
    })
    .collect()
}

fn read_non_bot_games(conn: &mut Connection) -> Option<Vec<PlaytakGame>> {
    let mut stmt = conn.prepare("SELECT id, date, size, player_white, player_black, notation, result, timertime, timerinc, rating_white, rating_black, unrated, tournament, komi, pieces, capstones FROM games
        WHERE NOT instr(player_white, \"Bot\") 
        AND NOT instr(player_black, \"Bot\") 
        AND NOT instr(player_white, \"bot\") 
        AND NOT instr(player_black, \"bot\")")
    .unwrap();
    let rows = stmt.query([]).unwrap().mapped(|row| {
        Ok(GameRow {
            id: row.get(0).unwrap(),
            date: DateTime::from_naive_utc_and_offset(
                DateTime::from_timestamp(row.get::<_, i64>(1).unwrap() / 1000, 0)
                    .unwrap()
                    .naive_local(),
                Utc,
            ),
            size: row.get(2).unwrap(),
            player_white: row.get(3).unwrap(),
            player_black: row.get(4).unwrap(),
            notation: row.get(5).unwrap(),
            result: row.get(6).unwrap(),
            timertime: Duration::from_secs(row.get(7).unwrap()),
            timerinc: Duration::from_secs(row.get(8).unwrap()),
            rating_white: row.get(9).unwrap(),
            rating_black: row.get(10).unwrap(),
            unrated: row.get(11).unwrap(),
            tournament: row.get(12).unwrap(),
            komi: row.get(13).unwrap(),
            pieces: row.get(14).unwrap(),
            capstones: row.get(15).unwrap(),
        })
    });
    rows.map(|row| PlaytakGame::try_from(row.unwrap()).ok())
        .collect()
}

fn generate_possible_puzzle<'a, const S: usize>(
    stats: &'a Stats,
    game: &'a PlaytakGame,
) -> impl Iterator<Item = PossiblePuzzle<S>> + 'a {
    assert_eq!(game.size, S);
    let mut position = <Position<S>>::start_position();
    let mut last_move_was_tinue = false;
    let moves: Vec<Move<S>> = game
        .notation
        .split_whitespace()
        .map(|mv| Move::from_string(mv).unwrap())
        .collect();

    moves.into_iter().filter_map(move |mv| {
        let last_tps = position.to_fen();

        position.do_move(mv);
        if position.game_result().is_some() {
            return None;
        }
        // Tinue is only possible after white has played S - 2 moves
        if position.half_moves_played() < (S - 2) * 2 {
            return None;
        }

        let tps = position.to_fen();
        let mut komi_position = position.clone();
        komi_position.set_komi(Komi::from_half_komi(4).unwrap());

        let tiltak_0komi_analysis_shallow = {
            let start_time = Instant::now();
            let result = tiltak_search(position.clone(), TILTAK_SHALLOW_NODES);

            stats.tiltak_non_tinue_short.record(start_time.elapsed());

            result
        };

        let tiltak_2komi_analysis_shallow = {
            let start_time = Instant::now();
            let result = tiltak_search(komi_position.clone(), TILTAK_SHALLOW_NODES);

            stats.tiltak_non_tinue_short.record(start_time.elapsed());

            result
        };

        let mut immediate_wins = vec![];
        position.generate_moves(&mut immediate_wins);
        immediate_wins.retain(|mv| {
            let reverse_move = position.do_move(*mv);
            let is_win = match position.game_result() {
                Some(GameResult::WhiteWin) if position.side_to_move() == Color::Black => true,
                Some(GameResult::BlackWin) if position.side_to_move() == Color::White => true,
                _ => false,
            };
            position.reverse_move(reverse_move);
            is_win
        });

        let topaz_result = topaz_search::<S>(&tps, stats);

        let mut possible_puzzle = PossiblePuzzle {
            playtak_game: game.clone(),
            tps: tps.clone(),
            previous_tps: last_tps,
            previous_move: mv,
            immediate_wins,
            followup_move: None,
            topaz_tinue: topaz_result.clone(),
            topaz_tinue_avoidance: None,
            tiltak_0komi_analysis_deep: None,
            tiltak_2komi_analysis_deep: None,
            last_move_was_tinue,
        };

        last_move_was_tinue = false;

        // If we have at least one half-decent move, check for tinue avoidance puzzle
        if (tiltak_0komi_analysis_shallow.score_first > 0.1
            || tiltak_2komi_analysis_shallow.score_first > 0.1)
            && !matches!(
                topaz_result,
                TopazResult::RoadWin | TopazResult::NonUniqueTinue(_)
            )
        {
            let start_time = Instant::now();
            let tinue_avoidance =
                topaz_tinue_avoidance(&mut position, &tiltak_0komi_analysis_shallow);
            stats.topaz_tinue_avoidance.record(start_time.elapsed());

            possible_puzzle.topaz_tinue_avoidance = Some(tinue_avoidance);
        };

        match topaz_result {
            TopazResult::NoTinue | TopazResult::AbortedFirst | TopazResult::AbortedSecond(_) => {
                if matches!(
                    possible_puzzle.topaz_tinue_avoidance,
                    Some(TopazAvoidanceResult::Defense(_))
                ) || tiltak_0komi_analysis_shallow.is_puzzle_candidate()
                    || tiltak_2komi_analysis_shallow.is_puzzle_candidate()
                {
                    let tiltak_start_time = Instant::now();
                    let tiltak_0komi_analysis_deep =
                        tiltak_search(position.clone(), TILTAK_DEEP_NODES);

                    possible_puzzle.tiltak_0komi_analysis_deep = Some(tiltak_0komi_analysis_deep);

                    let tiltak_2komi_analysis_deep =
                        tiltak_search(komi_position.clone(), TILTAK_DEEP_NODES);

                    possible_puzzle.tiltak_2komi_analysis_deep = Some(tiltak_2komi_analysis_deep);

                    stats
                        .tiltak_non_tinue_long
                        .record(tiltak_start_time.elapsed());
                }
            }
            TopazResult::RoadWin | TopazResult::NonUniqueTinue(_) => last_move_was_tinue = true,
            TopazResult::Tinue(_) => {
                let tiltak_start_time = Instant::now();
                let tiltak_0komi_analysis_deep = tiltak_search(position.clone(), TILTAK_DEEP_NODES);

                possible_puzzle.tiltak_0komi_analysis_deep = Some(tiltak_0komi_analysis_deep);

                let tiltak_2komi_analysis_deep =
                    tiltak_search(komi_position.clone(), TILTAK_DEEP_NODES);

                possible_puzzle.tiltak_2komi_analysis_deep = Some(tiltak_2komi_analysis_deep);

                stats.tiltak_tinue.record(tiltak_start_time.elapsed());

                last_move_was_tinue = true;
            }
        }
        Some(possible_puzzle)
    })
}

/*
CREATE TABLE "puzzles" (
    "game_id"	INTEGER NOT NULL,
    "tps"	TEXT NOT NULL UNIQUE,
    "solution"	TEXT NOT NULL,
    "tiltak_eval"	REAL NOT NULL,
    "tiltak_second_move_eval"	REAL NOT NULL,
    "tinue_length"	INTEGER,
    "is_tinue_avoidance"	INTEGER NOT NULL,
    PRIMARY KEY("tps")
);
*/

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Puzzle {
    game_id: u64,
    tps: String,
    solution: String,
    tiltak_0komi_eval: f32,
    tiltak_0komi_second_move_eval: f32,
    tiltak_0komi_pv_length: u32,
    tiltak_0komi_second_pv_length: u32,
    tiltak_2komi_eval: f32,
    tiltak_2komi_second_move_eval: f32,
    tiltak_2komi_pv_length: u32,
    tiltak_2komi_second_pv_length: u32,
    tinue_length: Option<u32>,
    tinue_avoidance_length: Option<u32>,
}

struct PossiblePuzzle<const S: usize> {
    playtak_game: PlaytakGame,
    tps: String,
    previous_tps: String,
    previous_move: Move<S>,
    immediate_wins: Vec<Move<S>>,
    followup_move: Option<Move<S>>,
    topaz_tinue: TopazResult<S>,
    topaz_tinue_avoidance: Option<TopazAvoidanceResult<S>>,
    tiltak_0komi_analysis_deep: Option<TiltakResult<S>>,
    tiltak_2komi_analysis_deep: Option<TiltakResult<S>>,
    last_move_was_tinue: bool,
}

impl<const S: usize> PossiblePuzzle<S> {
    fn make_real_puzzle(&self) -> Option<Puzzle> {
        let tiltak_0komi_eval = self.tiltak_0komi_analysis_deep.as_ref()?;
        let tiltak_2komi_eval = self.tiltak_2komi_analysis_deep.as_ref()?;
        let mut puzzle = Puzzle {
            game_id: self.playtak_game.id,
            tps: self.tps.clone(),
            solution: tiltak_0komi_eval
                .pv_first
                .first()
                .map(|mv| mv.to_string())
                .unwrap_or_default(),
            tiltak_0komi_eval: tiltak_0komi_eval.score_first,
            tiltak_0komi_second_move_eval: tiltak_0komi_eval.score_second,
            tiltak_0komi_pv_length: tiltak_0komi_eval.pv_first.len() as u32,
            tiltak_0komi_second_pv_length: tiltak_0komi_eval.pv_second.len() as u32,
            tiltak_2komi_eval: tiltak_2komi_eval.score_first,
            tiltak_2komi_second_move_eval: tiltak_2komi_eval.score_second,
            tiltak_2komi_pv_length: tiltak_2komi_eval.pv_first.len() as u32,
            tiltak_2komi_second_pv_length: tiltak_2komi_eval.pv_second.len() as u32,
            tinue_length: None,
            tinue_avoidance_length: None,
        };
        if let TopazResult::Tinue(moves) = &self.topaz_tinue {
            puzzle.tinue_length = Some(moves.len() as u32 + 1);
            puzzle.solution = moves[0].to_string();
        }
        if let Some(TopazAvoidanceResult::Defense(tinue_avoidance)) = &self.topaz_tinue_avoidance {
            puzzle.tinue_avoidance_length = Some(tinue_avoidance.longest_refutation_length + 2);
            puzzle.solution = tinue_avoidance.defense.to_string();
        }
        Some(puzzle)
    }
}

#[derive(Clone)]
struct TinueAvoidance<const S: usize> {
    defense: Move<S>,
    longest_refutation_length: u32,
}

#[derive(Default)]
pub struct Stats {
    topaz_tinue_first: TimeTracker,
    topaz_tinue_second: TimeTracker,
    topaz_tinue_avoidance: TimeTracker,
    tiltak_tinue: TimeTracker,
    tiltak_non_tinue_short: TimeTracker,
    tiltak_non_tinue_long: TimeTracker,
    desperado_defenses: TimeTracker,
    cataklysm: TimeTracker,
}

impl Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Topaz tinue first move: ")?;
        self.topaz_tinue_first.print_full();
        write!(f, "Topaz tinue second move: ")?;
        self.topaz_tinue_second.print_full();
        write!(f, "Topaz tinue avoidance: ")?;
        self.topaz_tinue_avoidance.print_full();
        write!(f, "Tiltak non tinue (short): ")?;
        self.tiltak_non_tinue_short.print_short();
        write!(f, "Tiltak non tinue (long): ")?;
        self.tiltak_non_tinue_long.print_short();
        write!(f, "Tiltak tinue: ")?;
        self.tiltak_tinue.print_short();
        write!(f, "Desperado defenses: ")?;
        self.desperado_defenses.print_short();
        write!(f, "Cataklysm: ")?;
        self.cataklysm.print_full();
        writeln!(f)
    }
}

#[derive(Default)]
struct TimeTracker {
    times: std::sync::Mutex<BTreeSet<Duration>>,
    total_time: std::sync::Mutex<Duration>,
}

impl TimeTracker {
    fn record(&self, time: Duration) {
        *self.total_time.lock().unwrap() += time;
        self.times.lock().unwrap().insert(time);
    }

    fn print_full(&self) {
        let times: Vec<Duration> = self.times.lock().unwrap().iter().cloned().collect();
        let total_time = self.total_time.lock().unwrap();
        println!(
            "{} events in {}s total time, average {:.2}s, longest {:.2}s, top 50% {:.2}s, top 1% {:.2}s, top 0.1% {:.2}s",
            times.len(),
            total_time.as_secs(),
            total_time.as_secs_f32() / times.len() as f32,
            times.last().cloned().unwrap_or_default().as_secs_f32(),
            times
                .get(times.len() / 2)
                .cloned()
                .unwrap_or_default()
                .as_secs_f32(),
            times
                .get(990 * times.len() / 1000)
                .cloned()
                .unwrap_or_default()
                .as_secs_f32(),
            times
                .get(999 * times.len() / 1000)
                .cloned()
                .unwrap_or_default()
                .as_secs_f32(),
        );
    }

    fn print_short(&self) {
        let times = self.times.lock().unwrap();
        let total_time = self.total_time.lock().unwrap();
        println!(
            "{} events in {}s total time, {:.2}s average time, longest {:.2}s",
            times.len(),
            total_time.as_secs(),
            total_time.as_secs_f32() / times.len() as f32,
            times
                .iter()
                .last()
                .cloned()
                .unwrap_or_default()
                .as_secs_f32()
        )
    }
}

#[derive(Debug, Clone)]
struct TiltakResult<const S: usize> {
    score_first: f32,
    pv_first: Vec<Move<S>>,
    score_second: f32,
    pv_second: Vec<Move<S>>,
}

impl<const S: usize> TiltakResult<S> {
    pub fn is_puzzle_candidate(&self) -> bool {
        self.score_first > 0.4
            && self.score_second < 0.6
            && self.score_first - self.score_second > 0.3
    }
}

fn tiltak_search<const S: usize>(position: Position<S>, nodes: u32) -> TiltakResult<S> {
    let settings1 = search::MctsSetting::default().arena_size_for_nodes(nodes);
    let mut tree1 = search::MonteCarloTree::new(position.clone(), settings1);
    for _ in 0..nodes {
        match tree1.select() {
            Ok(_) => (),
            Err(err) => {
                eprintln!("Tiltak search aborted early: {}", err);
                break;
            }
        }
    }
    let (best_move, score) = tree1.best_move().unwrap();

    let settings2 = search::MctsSetting::default()
        .arena_size_for_nodes(nodes)
        .exclude_moves(vec![best_move]);
    let mut tree2 = search::MonteCarloTree::new(position, settings2);
    for _ in 0..nodes {
        match tree2.select() {
            Ok(_) => (),
            Err(err) => {
                eprintln!("Tiltak search aborted early: {}", err);
                break;
            }
        }
    }

    // It's possible that the second move actually scores better than the first move
    // In that case, swap the moves, to make later processing easier
    if tree2.best_move().unwrap().1 > score {
        TiltakResult {
            score_first: tree2.best_move().unwrap().1,
            pv_first: tree2.pv().collect(),
            score_second: score,
            pv_second: tree1.pv().collect(),
        }
    } else {
        TiltakResult {
            score_first: score,
            pv_first: tree1.pv().collect(),
            score_second: tree2.best_move().unwrap().1,
            pv_second: tree2.pv().collect(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum TopazResult<const S: usize> {
    NoTinue,
    RoadWin,
    Tinue(Vec<Move<S>>),
    AbortedFirst,
    AbortedSecond(Vec<Move<S>>),
    NonUniqueTinue(Vec<Move<S>>),
}

impl<const S: usize> TopazResult<S> {
    pub const fn downcast_size<const N: usize>(self) -> TopazResult<N> {
        if S == N {
            unsafe { mem::transmute(self) }
        } else {
            panic!()
        }
    }
}

fn topaz_search<const S: usize>(tps: &str, stats: &Stats) -> TopazResult<S> {
    match S {
        5 => topaz_search_5s(tps, stats).downcast_size(),
        6 => topaz_search_6s(tps, stats).downcast_size(),
        _ => unimplemented!(),
    }
}

/// Run Topaz tinue search on the position, without checking tinue for the followup
/// Returns None on timeout, Some(false) if no tinue found
pub fn topaz_search_one_move<const S: usize>(tps: &str, stats: &Stats) -> Option<bool> {
    match S {
        5 => topaz_search_one_move_5s(tps, stats),
        6 => topaz_search_one_move_6s(tps, stats),
        _ => unimplemented!(),
    }
}

pub fn topaz_search_one_move_5s(tps: &str, stats: &Stats) -> Option<bool> {
    let board = topaz_tak::board::Board5::try_from_tps(tps).unwrap();
    let start_time = Instant::now();
    let mut first_tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
        .quiet()
        .limit(TOPAZ_FIRST_MOVE_NODES);
    let result = first_tinue_search.is_tinue();
    stats.topaz_tinue_first.record(start_time.elapsed());

    result
}

pub fn topaz_search_one_move_6s(tps: &str, stats: &Stats) -> Option<bool> {
    let board = topaz_tak::board::Board6::try_from_tps(tps).unwrap();
    let start_time = Instant::now();
    let mut first_tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
        .quiet()
        .limit(TOPAZ_FIRST_MOVE_NODES);
    let result = first_tinue_search.is_tinue();
    stats.topaz_tinue_first.record(start_time.elapsed());

    result
}

fn topaz_search_5s(tps: &str, stats: &Stats) -> TopazResult<5> {
    let board = topaz_tak::board::Board5::try_from_tps(tps).unwrap();
    let start_time = Instant::now();
    let mut first_tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
        .quiet()
        .limit(TOPAZ_FIRST_MOVE_NODES);
    let result = first_tinue_search.is_tinue();
    stats.topaz_tinue_first.record(start_time.elapsed());

    match result {
        None => TopazResult::AbortedFirst,
        Some(false) => TopazResult::NoTinue,
        Some(true) => {
            let pv = first_tinue_search.principal_variation();
            let tiltak_pv = pv
                .iter()
                .map(|mv| Move::from_string(mv.to_ptn::<Board5>().trim_end_matches('*')).unwrap())
                .collect();
            if !pv.is_empty() {
                let start_time = Instant::now();
                let mut second_tinue_search = topaz_tak::search::proof::TinueSearch::new(board)
                    .quiet()
                    .limit(TOPAZ_SECOND_MOVE_NODES)
                    .exclude(pv[0]);

                let result = second_tinue_search.is_tinue();
                stats.topaz_tinue_second.record(start_time.elapsed());

                match result {
                    Some(true) => TopazResult::NonUniqueTinue(tiltak_pv),
                    Some(false) => TopazResult::Tinue(tiltak_pv),
                    None => TopazResult::AbortedSecond(tiltak_pv),
                }
            } else {
                TopazResult::RoadWin
            }
        }
    }
}

fn topaz_search_6s(tps: &str, stats: &Stats) -> TopazResult<6> {
    let board = topaz_tak::board::Board6::try_from_tps(tps).unwrap();
    let start_time = Instant::now();
    let mut first_tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
        .quiet()
        .limit(TOPAZ_FIRST_MOVE_NODES);
    let is_tinue = first_tinue_search.is_tinue();
    stats.topaz_tinue_first.record(start_time.elapsed());

    if first_tinue_search.aborted() {
        return TopazResult::AbortedFirst;
    }
    if is_tinue == Some(true) {
        let pv = first_tinue_search.principal_variation();
        let tiltak_pv = pv
            .iter()
            .map(|mv| Move::from_string(mv.to_ptn::<Board6>().trim_end_matches('*')).unwrap())
            .collect();
        if !pv.is_empty() {
            let start_time = Instant::now();
            let mut second_tinue_search = topaz_tak::search::proof::TinueSearch::new(board)
                .quiet()
                .limit(TOPAZ_SECOND_MOVE_NODES)
                .exclude(pv[0]);
            second_tinue_search.is_tinue();
            stats.topaz_tinue_second.record(start_time.elapsed());

            if second_tinue_search.aborted() {
                TopazResult::AbortedSecond(tiltak_pv)
            } else if second_tinue_search.is_tinue() == Some(false) {
                TopazResult::Tinue(tiltak_pv)
            } else {
                TopazResult::NonUniqueTinue(tiltak_pv)
            }
        } else {
            TopazResult::RoadWin
        }
    } else {
        TopazResult::NoTinue
    }
}

#[derive(Clone)]
enum TopazAvoidanceResult<const S: usize> {
    Aborted,
    MultipleDefenses,
    NoDefense,
    Defense(TinueAvoidance<S>),
}

fn topaz_tinue_avoidance<const S: usize>(
    position: &mut Position<S>,
    shallow_tiltak_analysis: &TiltakResult<S>,
) -> TopazAvoidanceResult<S> {
    match S {
        5 => topaz_tinue_avoidance_5s(position, shallow_tiltak_analysis),
        6 => topaz_tinue_avoidance_6s(position, shallow_tiltak_analysis),
        _ => unimplemented!(),
    }
}

fn topaz_tinue_avoidance_5s<const S: usize>(
    position: &mut Position<S>,
    shallow_tiltak_analysis: &TiltakResult<S>,
) -> TopazAvoidanceResult<S> {
    let mut legal_moves = vec![];
    position.generate_moves(&mut legal_moves);
    // Check Tiltak's suggested moves first, to save time
    let index_first = legal_moves
        .iter()
        .position(|mv| *mv == shallow_tiltak_analysis.pv_first[0])
        .unwrap();
    let index_second = legal_moves
        .iter()
        .position(|mv| *mv == shallow_tiltak_analysis.pv_second[0])
        .unwrap();
    legal_moves.swap(0, index_first);
    legal_moves.swap(1, index_second);

    let mut has_aborted = false;
    let mut defense = None;
    let mut refutations = vec![];
    let mut longest_refutation_length = 0;

    for mv in legal_moves.iter() {
        let reverse_move = position.do_move(*mv);
        let board: Board5 = topaz_tak::board::Board5::try_from_tps(&position.to_fen()).unwrap();
        position.reverse_move(reverse_move);
        let mut tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
            .quiet()
            .limit(TOPAZ_AVOIDANCE_NODES);
        match tinue_search.is_tinue() {
            None => has_aborted = true, // If search aborts in one child, we can still conclude `MultipleDefenses` if two other children are not tinue
            Some(false) if defense.is_some() => return TopazAvoidanceResult::MultipleDefenses,
            Some(false) => defense = Some(mv),
            Some(true) => {
                if let Some(response) = tinue_search
                    .principal_variation()
                    .first()
                    .and_then(|mv| Move::from_string(&mv.to_ptn::<Board5>()).ok())
                {
                    refutations.push([*mv, response]);
                    longest_refutation_length = longest_refutation_length
                        .max(tinue_search.principal_variation().len() as u32)
                }
            }
        }
    }
    if has_aborted {
        TopazAvoidanceResult::Aborted
    } else if let Some(mv) = defense {
        TopazAvoidanceResult::Defense(TinueAvoidance {
            defense: *mv,
            longest_refutation_length,
        })
    } else {
        TopazAvoidanceResult::NoDefense
    }
}

fn topaz_tinue_avoidance_6s<const S: usize>(
    position: &mut Position<S>,
    shallow_tiltak_analysis: &TiltakResult<S>,
) -> TopazAvoidanceResult<S> {
    let mut legal_moves = vec![];
    position.generate_moves(&mut legal_moves);
    // Check Tiltak's suggested moves first, to save time
    let index_first = legal_moves
        .iter()
        .position(|mv| *mv == shallow_tiltak_analysis.pv_first[0])
        .unwrap();
    let index_second = legal_moves
        .iter()
        .position(|mv| *mv == shallow_tiltak_analysis.pv_second[0])
        .unwrap();
    legal_moves.swap(0, index_first);
    legal_moves.swap(1, index_second);

    let mut has_aborted = false;
    let mut defense = None;
    let mut refutations = vec![];
    let mut longest_refutation_length = 0;

    for mv in legal_moves.iter() {
        let reverse_move = position.do_move(*mv);
        let board: Board6 = topaz_tak::board::Board6::try_from_tps(&position.to_fen()).unwrap();
        position.reverse_move(reverse_move);
        let mut tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
            .quiet()
            .limit(TOPAZ_AVOIDANCE_NODES);
        match tinue_search.is_tinue() {
            None => has_aborted = true, // If search aborts in one child, we can still conclude `MultipleDefenses` if two other children are not tinue
            Some(false) if defense.is_some() => return TopazAvoidanceResult::MultipleDefenses,
            Some(false) => defense = Some(mv),
            Some(true) => {
                if let Some(response) = tinue_search
                    .principal_variation()
                    .first()
                    .and_then(|mv| Move::from_string(&mv.to_ptn::<Board6>()).ok())
                {
                    refutations.push([*mv, response]);
                    longest_refutation_length = longest_refutation_length
                        .max(tinue_search.principal_variation().len() as u32)
                }
            }
        }
    }
    if has_aborted {
        TopazAvoidanceResult::Aborted
    } else if let Some(mv) = defense {
        TopazAvoidanceResult::Defense(TinueAvoidance {
            defense: *mv,
            longest_refutation_length,
        })
    } else {
        TopazAvoidanceResult::NoDefense
    }
}

#[derive(Debug, Clone)]
struct PlaytakGame {
    id: u64,
    date_time: DateTime<Utc>,
    size: usize,
    player_white: String,
    player_black: String,
    notation: String,
    result_string: String,
    game_time: Duration,
    increment: Duration,
    rating_white: Option<i64>,
    rating_black: Option<i64>,
    is_rated: bool,
    is_tournament: bool,
    komi: Komi,
    flats: i64,
    caps: i64,
}

impl PlaytakGame {
    pub fn has_standard_piece_count(&self) -> bool {
        position::starting_stones(self.size) as i64 == self.flats
            && position::starting_capstones(self.size) as i64 == self.caps
    }

    pub fn is_bot_game(&self) -> bool {
        const BOTS: &[&str] = &[
            "TakticianBot",
            "alphatak_bot",
            "alphabot",
            "cutak_bot",
            "TakticianBotDev",
            "takkybot",
            "ShlktBot",
            "AlphaTakBot_5x5",
            "BeginnerBot",
            "TakkerusBot",
            "IntuitionBot",
            "AaaarghBot",
            "kriTakBot",
            "TakkenBot",
            "robot",
            "TakkerBot",
            "Geust93",
            "CairnBot",
            "VerekaiBot1",
            "BloodlessBot",
            "Tiltak_Bot",
            "Taik",
            "FlashBot",
            "FriendlyBot",
            "FPABot",
            "sTAKbot1",
            "sTAKbot2",
            "DoubleStackBot",
            "antakonistbot",
            "CrumBot",
        ];
        BOTS.contains(&self.player_white.as_str()) || BOTS.contains(&self.player_black.as_str())
    }

    pub fn game_is_legal(&self) -> bool {
        fn game_is_legal_sized<const S: usize>(notation: &str, komi: Komi) -> bool {
            let moves: Vec<Move<S>> = notation
                .split_whitespace()
                .map(|mv| Move::from_string(mv).expect(mv))
                .collect();
            let mut position: Position<S> = Position::start_position_with_komi(komi);
            let mut legal_moves = vec![];
            for mv in moves {
                if position.game_result().is_some() {
                    return false;
                }
                position.generate_moves(&mut legal_moves);
                if !legal_moves.contains(&mv) {
                    return false;
                }
                legal_moves.clear();
                position.do_move(mv);
            }
            true
        }
        match self.size {
            3 => game_is_legal_sized::<3>(&self.notation, self.komi),
            4 => game_is_legal_sized::<4>(&self.notation, self.komi),
            5 => game_is_legal_sized::<5>(&self.notation, self.komi),
            6 => game_is_legal_sized::<6>(&self.notation, self.komi),
            7 => game_is_legal_sized::<7>(&self.notation, self.komi),
            8 => game_is_legal_sized::<8>(&self.notation, self.komi),
            _ => unreachable!(),
        }
    }
}

impl TryFrom<GameRow> for PlaytakGame {
    type Error = ();
    fn try_from(row: GameRow) -> Result<Self, ()> {
        let notation = match row.size {
            3 => parse_notation::<3>(&row.notation),
            4 => parse_notation::<4>(&row.notation),
            5 => parse_notation::<5>(&row.notation),
            6 => parse_notation::<6>(&row.notation),
            7 => parse_notation::<7>(&row.notation),
            8 => parse_notation::<8>(&row.notation),
            _ => return Err(()),
        };
        Ok(PlaytakGame {
            id: row.id as u64,
            date_time: row.date,
            size: row.size as usize,
            player_white: row.player_white,
            player_black: row.player_black,
            notation,
            result_string: row.result,
            game_time: row.timertime,
            increment: row.timerinc,
            rating_white: if row.rating_white == 0 {
                None
            } else {
                Some(row.rating_white)
            },
            rating_black: if row.rating_black == 0 {
                None
            } else {
                Some(row.rating_black)
            },
            is_rated: !row.unrated,
            is_tournament: row.tournament,
            komi: Komi::from_half_komi(row.komi.try_into().map_err(|_| ())?).ok_or(())?,
            flats: if row.pieces == -1 {
                position::starting_stones(row.size as usize) as i64
            } else {
                row.pieces
            },
            caps: if row.capstones == -1 {
                position::starting_capstones(row.size as usize) as i64
            } else {
                row.capstones
            },
        })
    }
}

fn parse_notation<const S: usize>(notation: &str) -> String {
    if notation.is_empty() {
        String::new()
    } else {
        notation
            .split(',')
            .map(<Move<S>>::from_string_playtak)
            .fold(String::new(), |mut acc, mv| {
                write!(acc, " {}", mv).unwrap();
                acc
            })
    }
}

#[derive(Debug)]
struct GameRow {
    id: u32,
    date: DateTime<Utc>,
    size: u8,
    player_white: String,
    player_black: String,
    notation: String,
    result: String,
    timertime: Duration,
    timerinc: Duration,
    rating_white: i64,
    rating_black: i64,
    unrated: bool,
    tournament: bool,
    komi: u8,
    pieces: i64,
    capstones: i64,
}
