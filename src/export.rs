use board_game_traits::Position as PositionTrait;
use pgn_traits::PgnPosition;
use rand::seq::SliceRandom;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_rusqlite::{from_row, to_params_named};
use tiltak::position::{Komi, Move, Position};

use crate::{FinishedPuzzle, WinType};

#[derive(Serialize, Deserialize)]
struct ExportRow {
    root_tps: String,
    defender_start_move: String,
    size: usize,
    komi: String,
    player_white: String,
    player_black: String,
    solution: String,
    initial_rating: Option<i32>,
    rating: Option<i32>,
    target_time_seconds: u32,
    playtak_game_id: usize,
}

pub fn export_puzzles<const S: usize>(output_path: &str) {
    let puzzles_conn = Connection::open("puzzles.db").unwrap();
    let tinue_puzzles = read_tinue_puzzles(&puzzles_conn, S);
    let immediate_win_puzzles = read_immediate_win_puzzles::<S>(&puzzles_conn);

    println!(
        "Found {} tinue puzzles and {} immediate win puzzles",
        tinue_puzzles.len(),
        immediate_win_puzzles.len()
    );

    let output_conn = Connection::open(output_path).unwrap();

    output_conn
        .execute(
            "CREATE TABLE IF NOT EXISTS puzzles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_tps TEXT NOT NULL,
            defender_start_move TEXT NOT NULL,
            size INTEGER NOT NULL,
            komi TEXT NOT NULL,
            player_white TEXT NOT NULL,
            player_black TEXT NOT NULL,
            solution TEXT NOT NULL,
            initial_rating INTEGER,
            rating INTEGER,
            target_time_seconds INTEGER NOT NULL DEFAULT 60,
            playtak_game_id INTEGER NOT NULL,
            UNIQUE(root_tps, defender_start_move)
        )",
            [],
        )
        .unwrap();

    let mut export_puzzles = immediate_win_puzzles;
    for puzzle in tinue_puzzles {
        let mut export_puzzle = ExportRow {
            root_tps: puzzle.root_tps,
            defender_start_move: puzzle.defender_start_move,
            size: puzzle.size,
            komi: Komi::from_half_komi(puzzle.half_komi as i8)
                .unwrap()
                .to_string(),
            player_white: puzzle.player_white,
            player_black: puzzle.player_black,
            solution: puzzle.solution,
            initial_rating: None,
            rating: None,
            target_time_seconds: 60,
            playtak_game_id: puzzle.playtak_game_id as usize,
        };
        if export_puzzle.solution.split_whitespace().count() % 2 == 0 {
            // If the solution has an even amount of moves, the last move is an immediate win with more than one move
            export_puzzle.solution.push_str(" *");
        }
        export_puzzles.push(export_puzzle);
    }

    // We want the exported puzzles' IDs to be randomly distributed
    export_puzzles.shuffle(&mut rand::rng());

    for export_puzzle in export_puzzles {
        output_conn
            .execute(
                "INSERT INTO puzzles (root_tps, defender_start_move, size, komi, player_white, player_black, solution, initial_rating, rating, target_time_seconds, playtak_game_id)
                    VALUES (:root_tps, :defender_start_move, :size, :komi, :player_white, :player_black, :solution, :initial_rating, :rating, :target_time_seconds, :playtak_game_id)",
                to_params_named(&export_puzzle).unwrap().to_slice().as_slice()).unwrap();
    }
}

fn read_tinue_puzzles(puzzles_conn: &Connection, size: usize) -> Vec<FinishedPuzzle> {
    puzzles_conn
        .prepare("SELECT full_tinue_puzzles.*, games.player_white, games.player_black, games.size, games.komi as half_komi
            FROM full_tinue_puzzles
            JOIN games on full_tinue_puzzles.playtak_game_id = games.id WHERE games.size = ?1
            ORDER BY RANDOM()")
        .unwrap()
        .query_and_then([size], from_row::<FinishedPuzzle>)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImmediateWinPuzzle {
    pub size: usize,
    pub half_komi: i8,
    pub full_game_ptn: String,
    pub player_white: String,
    pub player_black: String,
    pub tps: String,
    pub tiltak_shallow_alternative_eval: f32,
    pub tiltak_deep_alternative_eval: Option<f32>,
    pub num_winning_moves: usize,
    pub num_winning_origin_squares: usize,
    pub win_type: WinType,
    pub playtak_game_id: i32,
}

fn read_immediate_win_puzzles<const S: usize>(puzzles_conn: &Connection) -> Vec<ExportRow> {
    let immediate_wins = puzzles_conn
        .prepare("SELECT games.size, games.komi AS half_komi, games.notation AS full_game_ptn, games.player_white, games.player_black, immediate_wins.*, games.id AS playtak_game_id
            FROM immediate_wins
            JOIN games on games.id = immediate_wins.game_id
            WHERE games.size = ?1 AND games.rating_white > 1200 AND games.rating_black > 1200 AND immediate_wins.tiltak_deep_alternative_eval < 0.5 AND immediate_wins.num_winning_origin_squares = 1
            ORDER BY RANDOM()
")
        .unwrap()
        .query_and_then([S], from_row::<ImmediateWinPuzzle>)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    immediate_wins
        .into_iter()
        .map(|puzzle| {
            let komi = Komi::from_half_komi(puzzle.half_komi).unwrap();
            let mut position: Position<S> = Position::start_position_with_komi(komi);
            for move_string in puzzle.full_game_ptn.split_whitespace() {
                let mv = Move::from_string(move_string).unwrap();
                let reverse_move = position.do_move(mv);
                if position.to_fen() == puzzle.tps {
                    // This is the position we want to export
                    let defender_start_move = mv;
                    position.reverse_move(reverse_move);
                    let tps = position.to_fen();
                    return ExportRow {
                        root_tps: tps,
                        defender_start_move: defender_start_move.to_string(),
                        size: puzzle.size,
                        komi: komi.to_string(),
                        player_white: puzzle.player_white.clone(),
                        player_black: puzzle.player_black.clone(),
                        solution: "*".to_string(),
                        initial_rating: None,
                        rating: None,
                        target_time_seconds: 60,
                        playtak_game_id: puzzle.playtak_game_id as usize,
                    };
                }
            }
            unreachable!("Could not find the position in the game PTN")
        })
        .collect()
}
