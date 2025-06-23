use std::{sync::atomic::Ordering, time::Instant};

use board_game_traits::{Color, Position as PositionTrait};
use cataklysm::{
    game::{Eval, Options},
    new_game,
};
use pgn_traits::PgnPosition;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;
use serde_rusqlite::from_rows;
use tiltak::position::{Komi, Position};

use crate::{NUM_GAMES_PROCESSED, PuzzleRoot, Stats};

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

pub fn find_potential_gaelet<const S: usize>(db_path: &str) {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    let mut stmt = conn
        .prepare("SELECT games.id AS playtak_game_id, games.komi AS half_komi, puzzles.* FROM puzzles JOIN games ON puzzles.game_id = games.id
            WHERE games.size = ?1 AND tinue_length IS NULL AND tinue_avoidance_length IS NULL AND (games.komi = 0 AND tiltak_0komi_eval > 0.8 AND tiltak_0komi_eval - tiltak_0komi_second_move_eval > 0.3 OR games.komi = 4 AND tiltak_2komi_eval > 0.8 AND tiltak_2komi_eval - tiltak_2komi_second_move_eval > 0.3)")
        .unwrap();

    let gaelet_roots: Vec<GaeletRoot<S>> = from_rows::<GaeletRoot<S>>(stmt.query([S]).unwrap())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("Found {} potential Gaelet roots", gaelet_roots.len());

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

    endgame_gaelet_roots.into_par_iter().for_each(|root| {
        let komi = Komi::from_half_komi(root.half_komi).unwrap();
        let position = Position::<S>::from_fen_with_komi(&root.tps, komi).unwrap();
        let eval = cataklysm_search_root(position, &root, 13, &stats);
        if eval.is_decisive() {
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
) -> Eval {
    let mut options: Options = Options {
        half_komi: position.komi().half_komi() as i32,
        ..Options::default(S).unwrap()
    };

    options.params.tt_size = 1 << 26; // Set the transposition table size to 2 GiB

    let mut game = new_game(S, options).unwrap();
    game.set_position(&position.to_fen()).unwrap();

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
    eval
}
