use std::{
    io,
    sync::{Mutex, atomic::Ordering},
    time::Instant,
};

use board_game_traits::{Color, Position as PositionTrait};
use cataklysm::{
    game::{Eval, Options},
    new_game,
};
use pgn_traits::PgnPosition;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_rusqlite::{from_rows, to_params_named};
use tiltak::position::{Komi, Move, Position};

use crate::{
    NUM_GAMES_PROCESSED, PuzzleRoot, Stats, TILTAK_DEEP_NODES, TiltakResult,
    find_last_defending_move, tiltak_search,
};

#[derive(Debug, Deserialize)]
pub struct GaeletRoot<const S: usize> {
    playtak_game_id: u32,
    half_komi: i8,
    tps: String,
    tiltak_0komi_eval: f32,
    tiltak_0komi_second_move_eval: f32,
    tiltak_2komi_eval: f32,
    tiltak_2komi_second_move_eval: f32,
}

pub fn reserves_left_for_us<const S: usize>(position: &Position<S>) -> u8 {
    if position.side_to_move() == Color::White {
        position.white_reserves_left() + position.white_caps_left()
    } else {
        position.black_reserves_left() + position.black_caps_left()
    }
}

#[derive(Serialize, Deserialize)]
struct FullGaeletPuzzle<const S: usize> {
    playtak_game_id: u32,
    tps: String,
    goes_to_win: bool,
    solution: String,
}

pub fn find_potential_gaelet<const S: usize>(db_path: &str) {
    let conn = Connection::open(db_path).unwrap();
    let mut stmt = conn
        .prepare("SELECT games.id AS playtak_game_id, games.komi AS half_komi, puzzles.* FROM puzzles JOIN games ON puzzles.game_id = games.id
            WHERE games.size = ?1 AND tinue_length IS NULL AND tinue_avoidance_length IS NULL AND (games.komi = 0 AND tiltak_0komi_eval > 0.8 AND tiltak_0komi_eval - tiltak_0komi_second_move_eval > 0.3 OR games.komi = 4 AND tiltak_2komi_eval > 0.8 AND tiltak_2komi_eval - tiltak_2komi_second_move_eval > 0.3)")
        .unwrap();

    let gaelet_roots: Vec<GaeletRoot<S>> = from_rows::<GaeletRoot<S>>(stmt.query([S]).unwrap())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("Found {} potential Gaelet roots", gaelet_roots.len());
    conn.execute(
        "CREATE TABLE IF NOT EXISTS full_gaelet_puzzles (
            playtak_game_id	INTEGER NOT NULL,
            tps	TEXT PRIMARY KEY,
            goes_to_win INTEGER NOT NULL,
            solution TEXT NOT NULL,
            FOREIGN KEY(playtak_game_id) REFERENCES games(id)
            )",
        [],
    )
    .unwrap();

    // Positions where the side to move has 4 or fewer reserves left
    let endgame_gaelet_roots: Vec<GaeletRoot<S>> = gaelet_roots
        .into_iter()
        .filter(|root| {
            let komi = Komi::from_half_komi(root.half_komi).unwrap();
            let position = Position::<S>::from_fen_with_komi(&root.tps, komi).unwrap();
            reserves_left_for_us(&position) <= 4
        })
        .collect();
    let num_endgame_roots = endgame_gaelet_roots.len();
    println!("Found {} endgame Gaelet roots", num_endgame_roots);
    let start_time = Instant::now();
    let stats = Stats::default();

    let num_wins_found = std::sync::atomic::AtomicUsize::new(0);

    let manual_eval_mutex = Mutex::new(());

    endgame_gaelet_roots.into_par_iter().for_each_init(|| Connection::open(db_path).unwrap(), |conn, root| {
        let komi = Komi::from_half_komi(root.half_komi).unwrap();
        let mut position = Position::<S>::from_fen_with_komi(&root.tps, komi).unwrap();
        let (eval, pv) = cataklysm_search(position.clone(), 11, &stats);
        if eval.is_decisive() {
            let solution = Move::from_string(pv.split_whitespace().next().unwrap()).unwrap();
            assert!(position.move_is_legal(solution));
            let tiltak_eval = tiltak_search(position.clone(), TILTAK_DEEP_NODES);
            position.do_move(solution);
            if position.game_result().is_some() {
                // Skip positions that are already terminal
                return;
            }
            let moves = vec![(solution, Some((eval, tiltak_eval)))];
            let mut solutions = vec![(moves.clone(), false)];

            // println!(
            //     "Finding Gaelet followup for position: {}",
            //     position.to_fen()
            // );
            find_gaelet_followup(&mut position, &moves, &mut solutions);
            let _guard = manual_eval_mutex.lock().unwrap();
            println!(
                "Found {} Gaelet followup solutions to length {} in #{} gaelet on {}",
                solutions.len(),
                pv.split_whitespace().count(),
                root.playtak_game_id,
                position.to_fen()
            );
            for (solution, _goes_to_win) in solutions.iter() {
                println!(
                    "Gaelet solution: {}",
                    solution
                        .iter()
                        .map(|(mv, eval)| format!(
                            "{} {}",
                            mv,
                            eval.as_ref()
                                .map(|(cata_eval, tiltak_eval)| format!(
                                    "({}, tiltak 1st {} {:.3}, 2nd {} {:.3})",
                                    cata_eval,
                                    tiltak_eval.pv_first[0],
                                    tiltak_eval.score_first,
                                    tiltak_eval.pv_second[0],
                                    tiltak_eval.score_second
                                ))
                                .unwrap_or("".to_string())
                        ))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }

            println!("https://playtak.com/games/{}/ninjaviewer", root.playtak_game_id);
            let solution_strings = solutions
                .iter()
                .map(|(solution, goes_to_win)| {
                    let mut move_string = solution
                        .iter()
                        .map(|(mv, _)| mv.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    // Last move is a wildcard move, i.e. many moves win
                    if solution.len() % 2 == 0 {
                        move_string.push_str(" *");
                    }
                    print!("{}", move_string);
                    if *goes_to_win {
                        println!(" (goes to win)");
                    } else {
                        println!();
                    }
                    (move_string, goes_to_win)
                })
                .collect::<Vec<_>>();
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
                    let solution_index = solution_strings.iter().position(|(s, _)| s == input);
                    let Some(solution_index) = solution_index else {
                        println!("Solution not found in the list, try again");
                        continue;
                    };

                    let full_puzzle: FullGaeletPuzzle<S> =
                        FullGaeletPuzzle {
                            playtak_game_id: root.playtak_game_id,
                            tps: position.to_fen(),
                            goes_to_win: *solution_strings[solution_index].1,
                            solution: solution_strings[solution_index].0.clone(),
                        };

                    let rows_affected = conn.execute("INSERT OR IGNORE INTO full_gaelet_puzzles VALUES (:playtak_game_id, :tps, :goes_to_win, :solution)", to_params_named(&full_puzzle).unwrap().to_slice().as_slice()).unwrap();
                    if rows_affected != 1 {
                        println!("Warning: {} rows affected by insert", rows_affected);
                    }
                    break;
                }

            num_wins_found.fetch_add(1, Ordering::Relaxed);
        }
        let n = NUM_GAMES_PROCESSED.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 10 == 0 {
            println!(
                "Processed {}/{} positions, found {}/{} wins in {:.1}s",
                n,
                num_endgame_roots,
                num_wins_found.load(Ordering::Relaxed),
                n,
                start_time.elapsed().as_secs_f64()
            );
            println!();
        }
    });
}

pub fn find_gaelet_followup<const S: usize>(
    position: &mut Position<S>,
    moves: &[(Move<S>, Option<(Eval, TiltakResult<S>)>)],
    solutions: &mut Vec<(Vec<(Move<S>, Option<(Eval, TiltakResult<S>)>)>, bool)>,
) {
    // Special logic if the defender is losing next move no matter what,
    // so that the solution is always extended up until a win
    if let Some((defender_move, winning_move)) = find_last_defending_move(position) {
        let mut new_solution = moves.to_vec();
        new_solution.push((defender_move, None));
        if let Some(winning_move) = winning_move {
            let reverse_move = position.do_move(defender_move);
            let tiltak_eval = tiltak_search(position.clone(), TILTAK_DEEP_NODES);
            let (child_eval, _) = cataklysm_search(position.clone(), 3, &Stats::default());

            new_solution.push((winning_move, Some((child_eval, tiltak_eval.clone()))));

            position.reverse_move(reverse_move);
        }
        solutions.push((new_solution.clone(), true));
        return;
    }
    let mut defender_moves = vec![];
    position.generate_moves(&mut defender_moves);
    for defender_move in defender_moves {
        let reverse_move = position.do_move(defender_move);
        if position.game_result().is_some() {
            // Skip positions that are already terminal
            position.reverse_move(reverse_move);
            continue;
        }
        let tiltak_eval = tiltak_search(position.clone(), TILTAK_DEEP_NODES);
        let tiltak_delta = tiltak_eval.score_first - tiltak_eval.score_second;
        if tiltak_delta > 0.1 || tiltak_eval.score_first < 0.9 {
            let mut new_solution = moves.to_vec();
            new_solution.push((defender_move, None));
            let (child_eval, _) = cataklysm_search(position.clone(), 11, &Stats::default());
            new_solution.push((
                tiltak_eval.pv_first[0],
                Some((child_eval, tiltak_eval.clone())),
            ));
            solutions.push((new_solution.clone(), false));

            let reverse_attacker_move = position.do_move(tiltak_eval.pv_first[0]);
            if position.game_result().is_none() {
                find_gaelet_followup(position, &new_solution, solutions);
            }
            position.reverse_move(reverse_attacker_move);
        }
        position.reverse_move(reverse_move);
    }
}

pub fn analyze_puzzle_cataklysm<const S: usize>(puzzle: &PuzzleRoot<S>) -> (Eval, String) {
    let position = Position::<S>::from_fen(&puzzle.tps).unwrap();
    let max_depth = 9;

    let mut options: Options = Options {
        half_komi: position.komi().half_komi() as i32,
        ..Options::default(S).unwrap()
    };

    options.params.tt_size = 1 << 24; // Set the transposition table size to 512 MiB

    let mut game = new_game(S, options).unwrap();
    game.set_position(&position.to_fen()).unwrap();

    // let start_time = Instant::now();
    let mut eval = Eval::ZERO;

    for depth in 1..=max_depth {
        (eval, _) = game.search(depth).unwrap();
    }

    // let nodes = game.nodes();
    let pv = game.pv();

    // println!(
    //     "Depth: {}, nodes: {}, eval: {}, move: {}, pv: {}, {:.2}s elapsed",
    //     depth,
    //     nodes,
    //     eval,
    //     mv,
    //     pv,
    //     start_time.elapsed().as_secs_f64()
    // );
    (eval, pv.to_string())
}
pub fn cataklysm_search<const S: usize>(
    position: Position<S>,
    max_depth: u32,
    stats: &Stats,
) -> (Eval, String) {
    let mut options: Options = Options {
        half_komi: position.komi().half_komi() as i32,
        ..Options::default(S).unwrap()
    };

    options.params.tt_size = 1 << 20; // Set the transposition table size to 32 MiB

    let mut game = new_game(S, options).unwrap();
    game.set_position(&position.to_fen()).unwrap();
    let start_time = Instant::now();

    for depth in 1..=max_depth {
        let (eval, _mv) = game.search(depth).unwrap();
        if depth == max_depth {
            let pv = game.pv();
            stats.cataklysm.record(start_time.elapsed());
            return (eval, pv.to_string());
        }
    }
    unreachable!()
}

pub fn cataklysm_search_root<const S: usize>(
    position: Position<S>,
    gaelet_root: &GaeletRoot<S>,
    max_depth: u32,
    stats: &Stats,
) -> (Eval, String) {
    let start_time = Instant::now();

    let (eval, pv) = cataklysm_search(position.clone(), max_depth, stats);

    println!(
        "#{}: Komi: {}, tps: {}",
        gaelet_root.playtak_game_id,
        position.komi(),
        position.to_fen()
    );
    if gaelet_root.half_komi == 0 {
        print!(
            "Tiltak 0 komi eval: {:.3}, second move eval: {:.3}",
            gaelet_root.tiltak_0komi_eval, gaelet_root.tiltak_0komi_second_move_eval
        );
    } else {
        print!(
            "Tiltak 2 komi eval: {:.3}, second move eval: {:.3}",
            gaelet_root.tiltak_2komi_eval, gaelet_root.tiltak_2komi_second_move_eval
        );
    }
    println!(", reserves left: {}", reserves_left_for_us(&position));
    println!(
        "Depth: {}, eval: {}, pv: {}, {:.2}s elapsed",
        max_depth,
        eval,
        pv,
        start_time.elapsed().as_secs_f64()
    );

    println!();
    (eval, pv)
}
