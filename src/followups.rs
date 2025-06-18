use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use board_game_traits::{GameResult, Position as PositionTrait};
use pgn_traits::PgnPosition;
use rayon::prelude::*;
use rusqlite::Connection;
use tiltak::position::{Komi, Move, Position};

use crate::{
    NUM_GAMES_PROCESSED, PuzzleF, PuzzleRoot, Stats, TILTAK_DEEP_NODES, TinueFollowup, TopazResult,
    find_last_defending_move, tiltak_search, topaz_search,
};

pub fn find_all_from_db<const S: usize>(db_path: &str) {
    let puzzles_conn = Connection::open(db_path).unwrap();
    let mut stmt = puzzles_conn.prepare("SELECT puzzles.tps, puzzles.solution, puzzles.tinue_length, games.id FROM puzzles JOIN games ON puzzles.game_id = games.id
        WHERE games.size = ?1 AND puzzles.tinue_length NOT NULL AND (tiltak_0komi_second_move_eval < 0.7 OR tiltak_2komi_second_move_eval < 0.7) ORDER BY RANDOM()")
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
    let candidates = find_all(&puzzle_roots);
    println!("Found {} candidates", candidates.len());
}

pub fn find_all<const S: usize>(puzzle_roots: &[PuzzleRoot<S>]) -> Vec<TinuePuzzleCandidate<S>> {
    let stats = Arc::new(Stats::default());

    println!(
        "Got {} puzzles roots, like {}",
        puzzle_roots.len(),
        puzzle_roots
            .first()
            .map(ToString::to_string)
            .unwrap_or_default()
    );

    let start_time = Instant::now();
    let num_root_puzzles = puzzle_roots.len();

    let num_goes_to_road = AtomicU64::new(0);
    let num_single_solution_to_road = AtomicU64::new(0);
    let num_single_solution = AtomicU64::new(0);

    let puzzle_candidates = puzzle_roots
        .par_iter()
        .map(|puzzle_root| {
            let mut position = Position::from_fen(&puzzle_root.tps).unwrap();
            let mv = puzzle_root.solution;
            assert!(position.move_is_legal(mv));
            position.do_move(mv);

            let tinue_candidate = extract_possible_full_tinues(position, &stats);

            if tinue_candidate.solutions.is_empty() {
                println!("No solutions found for puzzle root {}", puzzle_root);
            }
            if tinue_candidate
                .solutions
                .iter()
                .any(|(_, is_road)| *is_road)
            {
                num_goes_to_road.fetch_add(1, Ordering::Relaxed);
                if tinue_candidate
                    .solutions
                    .iter()
                    .filter(|(_, is_road)| *is_road)
                    .count()
                    == 1
                {
                    num_single_solution_to_road.fetch_add(1, Ordering::Relaxed);
                }
            }
            if tinue_candidate.solutions.len() == 1 {
                num_single_solution.fetch_add(1, Ordering::Relaxed);
            }

            let n = NUM_GAMES_PROCESSED.fetch_add(1, Ordering::AcqRel);

            println!(
                "{}/{} puzzles processed in {:.1}s, ETA {:.1}s, results for {}:",
                n,
                num_root_puzzles,
                start_time.elapsed().as_secs_f32(),
                (start_time.elapsed().as_secs_f32() / n as f32)
                    * (num_root_puzzles as f32 - n as f32),
                puzzle_root
            );
            for (tinue, goes_to_road) in tinue_candidate.solutions.iter() {
                print!(
                    "Goes to road: {}, solution: {}",
                    goes_to_road,
                    tinue
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                if tinue.len() % 2 == 0 {
                    println!(" *");
                } else {
                    println!();
                }
            }
            println!();

            if (n + 1) % 10 == 0 {
                println!(
                    "{}/{} puzzles go to a road, {}/{} goes to road with single solution, {}/{} puzzles have a single solution",
                    num_goes_to_road.load(Ordering::Relaxed),
                    n + 1,
                    num_single_solution_to_road.load(Ordering::Relaxed),
                    n + 1,
                    num_single_solution.load(Ordering::Relaxed),
                    n + 1,
                );
                println!("Time usage stats:");
                println!("{}", stats);
            }
            tinue_candidate
        })
        .collect();
    puzzle_candidates
}

pub struct TinuePuzzleCandidate<const S: usize> {
    pub position: Position<S>,
    pub solutions: Vec<(Vec<Move<S>>, bool)>,
}

pub fn extract_possible_full_tinues<const S: usize>(
    mut position: Position<S>,
    stats: &Stats,
) -> TinuePuzzleCandidate<S> {
    let mut moves = vec![position.moves().last().expect("No last move found").clone()];
    let mut possible_lines = vec![];

    find_followup_recursive(&mut position, &mut moves, &stats, &mut possible_lines);

    let tinue = TinuePuzzleCandidate {
        position: position.clone(),
        solutions: possible_lines.clone(),
    };
    tinue
}

fn find_followup_recursive<const S: usize>(
    position: &mut Position<S>,
    moves: &mut Vec<Move<S>>,
    stats: &Stats,
    possible_lines: &mut Vec<(Vec<Move<S>>, bool)>,
) {
    // Check if we're one move (two ply) away from a road win
    // If so, return early
    if let Some((last_move, unique_win)) = find_last_defending_move(position) {
        moves.push(last_move);
        if let Some(unique_move) = unique_win {
            moves.push(unique_move);
        }
        possible_lines.push((moves.clone(), true));
        if unique_win.is_some() {
            moves.pop();
        }
        moves.pop();
        return;
    }

    let followups = find_followup::<S>(position.clone(), stats);

    // We know that the position is tinue, but not a 2-ply win, so ignore those
    let followups = followups
        .into_iter()
        .filter_map(|followup| match followup {
            PuzzleF::UniqueTinue(tinue) => Some(tinue),
            PuzzleF::NonUniqueTinue | PuzzleF::UniqueRoadWin(_, _) | PuzzleF::NonUniqueRoadWin => {
                None
            }
        })
        .collect::<Vec<_>>();

    if followups.is_empty() {
        possible_lines.push((moves.clone(), false));
        return;
    }
    for followup in followups {
        let reverse_move = position.do_move(followup.parent_move);
        let reverse_move2 = position.do_move(followup.solution);
        moves.push(followup.parent_move);
        moves.push(followup.solution);

        find_followup_recursive(position, moves, stats, possible_lines);

        moves.pop();
        moves.pop();

        position.reverse_move(reverse_move2);
        position.reverse_move(reverse_move);
    }
}

pub fn find_followup<const S: usize>(mut position: Position<S>, stats: &Stats) -> Vec<PuzzleF<S>> {
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
