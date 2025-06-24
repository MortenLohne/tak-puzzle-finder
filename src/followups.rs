use std::{
    collections::HashMap,
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
use serde::{Deserialize, Serialize};
use serde_rusqlite::{from_row, to_params_named};
use tiltak::position::{ExpMove, Komi, Move, Piece, Position, Role};

use crate::{
    NUM_GAMES_PROCESSED, PuzzleF, PuzzleRoot, Stats, TILTAK_DEEP_NODES, TinueFollowup, TopazResult,
    find_last_defending_move, find_last_defending_move_among_moves, tiltak_search, topaz_search,
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
    let candidates = find_all(&puzzle_roots, db_path);
    println!("Found {} candidates", candidates.len());
}

fn insert_puzzle_candidate<const S: usize>(
    conn: &Connection,
    game_id: u64,
    puzzle_candidate: &TinuePuzzleCandidate2<S>,
    root_tps: &str,
) {
    let puzzle_line_rows =
        TinueLineCandidateRow::from_puzzle_candidate(puzzle_candidate.clone(), game_id, root_tps);

    let mut stmt = conn
        .prepare(
            "INSERT OR IGNORE INTO candidate_full_solutions
        VALUES (:game_id, :tps, :solution, :goes_to_road, :pure_recaptures_end_sequence, :trivial_desperado_defense_skipped)",
        )
        .unwrap();

    for solution in puzzle_line_rows {
        stmt.execute(to_params_named(&solution).unwrap().to_slice().as_slice())
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to insert puzzle candidate: tps {} for game_id {}: {}",
                    solution.tps, game_id, err
                );
            });
    }
}

pub fn evaluate_followups<const S: usize>(db_path: &str) {
    let conn = Connection::open(db_path).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT candidate_full_solutions.* FROM candidate_full_solutions
            JOIN games ON candidate_full_solutions.game_id = games.id
            WHERE games.size = ?1",
        )
        .unwrap();
    let followups: Vec<TinueLineCandidateRow> = stmt
        .query_and_then([S], from_row::<TinueLineCandidateRow>)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let followups_per_position: HashMap<String, Vec<TinueLineCandidateRow>> = followups
        .into_iter()
        .fold(HashMap::new(), |mut acc, followup| {
            acc.entry(followup.tps.clone()).or_default().push(followup);
            acc
        });

    let candidates: Vec<TinuePuzzleCandidate2<S>> = followups_per_position
        .iter()
        .map(|(tps, followups)| TinuePuzzleCandidate2 {
            position: Position::from_fen(tps).unwrap(),
            solutions: followups
                .iter()
                .map(|followup| TinueLineCandidate {
                    moves: followup
                        .solution
                        .split_whitespace()
                        .map(|s| Move::from_string(s).unwrap())
                        .collect(),
                    goes_to_road: followup.goes_to_road,
                    pure_recaptures_end_sequence: followup
                        .pure_recaptures_end_sequence
                        .split_whitespace()
                        .map(|s| Move::from_string(s).unwrap())
                        .collect(),
                    trivial_desperado_defense_skipped: followup
                        .trivial_desperado_defense_skipped
                        .as_ref()
                        .map(|s| {
                            s.split_whitespace()
                                .map(|s| Move::from_string(s).unwrap())
                                .collect()
                        }),
                })
                .collect(),
        })
        .collect();

    println!("Reanalyzing {} candidates", candidates.len());
    // Reanalyze all candidates, if we've changed the puzzle analysis since it was written to the database
    let candidates = candidates
        .into_par_iter()
        .map(|candidate| candidate.reanalyze())
        .collect::<Vec<_>>();
    println!("Reanalyzed {} candidates", candidates.len());

    let mut num_approved = 0;
    let mut num_denied = 0;
    let mut num_manual_review_single_solution = 0;
    let mut num_manual_review_multi_solution = 0;
    let mut num_manual_goes_to_road = 0;
    let mut num_manual_no_road = 0;
    for candidate in candidates.iter() {
        let puzzle_evaluation = evaluate_puzzle_candidate(candidate.clone());

        match puzzle_evaluation {
            PuzzleCandidateEvaluation::Approve(_) => {
                num_approved += 1;
            }
            PuzzleCandidateEvaluation::Deny => {
                num_denied += 1;
            }
            PuzzleCandidateEvaluation::ManualReview(_) if candidate.solutions.len() == 1 => {
                num_manual_review_single_solution += 1;
            }
            PuzzleCandidateEvaluation::ManualReview(_) => {
                num_manual_review_multi_solution += 1;
            }
        }
        if matches!(
            puzzle_evaluation,
            PuzzleCandidateEvaluation::ManualReview(_)
        ) {
            if candidate
                .solutions
                .iter()
                .any(|s| s.goes_to_road || !s.pure_recaptures_end_sequence.is_empty())
            {
                num_manual_goes_to_road += 1;
            } else {
                num_manual_no_road += 1;
            }
        }
        println!("Position: {}", candidate.position.to_fen());
        for processed_candidate in candidate.solutions.iter() {
            print!(
                "Goes to road: {}, solution: {}",
                processed_candidate.goes_to_road,
                processed_candidate
                    .moves
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if processed_candidate.moves.len() % 2 == 0 {
                print!(" *");
            }
            if !processed_candidate.pure_recaptures_end_sequence.is_empty() {
                print!(
                    " (desperado defense: {})",
                    processed_candidate
                        .pure_recaptures_end_sequence
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            if let Some(skipped_line) = &processed_candidate.trivial_desperado_defense_skipped {
                print!(
                    " (trivial desperado defense skipped: {})",
                    skipped_line
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            if puzzle_evaluation == PuzzleCandidateEvaluation::Approve(processed_candidate.clone())
            {
                print!(" (approved solution)");
            } else if puzzle_evaluation == PuzzleCandidateEvaluation::Deny {
                print!(" (denied solution)");
            } else if puzzle_evaluation
                == PuzzleCandidateEvaluation::ManualReview(processed_candidate.clone())
            {
                print!(" (primary candidate solution)");
            }
            println!();
        }
        println!();
    }

    println!(
        "Evaluated {} candidates: {} approved, {} denied, {} manual that goes to road or pure recapture sequence, {} manual that doesn't, {} manual review single solution, {} manual review multi solution",
        candidates.len(),
        num_approved,
        num_denied,
        num_manual_goes_to_road,
        num_manual_no_road,
        num_manual_review_single_solution,
        num_manual_review_multi_solution
    );
}

pub fn find_all<const S: usize>(
    puzzle_roots: &[PuzzleRoot<S>],
    db_path: &str,
) -> Vec<TinuePuzzleCandidate<S>> {
    let stats = Arc::new(Stats::default());

    // Create a new table for storing the puzzle candidates
    let conn = Connection::open(db_path).unwrap();
    conn.execute(
        "CREATE TABLE IF NOT EXISTS candidate_full_solutions (
            game_id INTEGER NOT NULL,
            tps TEXT NOT NULL,
            solution TEXT NOT NULL,
            goes_to_road BOOLEAN NOT NULL,
            pure_recaptures_end_sequence TEXT NOT NULL,
            trivial_desperado_defense_skipped TEXT,
            FOREIGN KEY (tps) REFERENCES puzzles(tps)
        )",
        [],
    )
    .unwrap();

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
    let num_one_move_solution_and_no_desperado = AtomicU64::new(0);
    let num_approved = AtomicU64::new(0);
    let num_denied = AtomicU64::new(0);
    let num_manual_review = AtomicU64::new(0);

    let puzzle_candidates = puzzle_roots
        .par_iter()
        .map_init(||Connection::open(db_path).unwrap(), |conn, puzzle_root| {
            let mut position = Position::from_fen(&puzzle_root.tps).unwrap();
            let root_position = position.clone();
            let mv = puzzle_root.solution;
            assert!(position.move_is_legal(mv));
            position.do_move(mv);

            let tinue_candidate = extract_possible_full_tinues(root_position, mv, &stats);
            let desperado_followup_start_time = Instant::now();
            let processed_candidate = process_full_tinue(tinue_candidate.clone());
            stats
                .desperado_defenses
                .record(desperado_followup_start_time.elapsed());

            insert_puzzle_candidate(conn, puzzle_root.playtak_game_id as u64, &processed_candidate, &puzzle_root.tps);

            let desperado_defenses = tinue_candidate
                .solutions
                .iter()
                .map(|(tinue, goes_to_road)| {
                    if !goes_to_road {
                        let mut position_clone = position.clone();
                        for mv in tinue.iter() {
                            assert!(position_clone.move_is_legal(*mv));
                            position_clone.do_move(*mv);
                        }
                        let desperado_start_time = Instant::now();
                        let result = find_desperado_defense_lines(&mut position_clone);
                        stats
                            .desperado_defenses
                            .record(desperado_start_time.elapsed());
                        result
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            if tinue_candidate.solutions.is_empty() {
                println!("No solutions found for puzzle root {}", puzzle_root);
            }
            let any_solution_goes_to_road = tinue_candidate
                .solutions
                .iter()
                .any(|(_, is_road)| *is_road);
            if any_solution_goes_to_road
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
                if tinue_candidate.solutions[0].0.len() == 1 && desperado_defenses[0].is_none() {
                    num_one_move_solution_and_no_desperado.fetch_add(1, Ordering::Relaxed);
                }
            }

            let n = NUM_GAMES_PROCESSED.fetch_add(1, Ordering::AcqRel) + 1;

            println!(
                "{}/{} puzzles processed in {:.1}s, ETA {:.1}s, results for {}:",
                n,
                num_root_puzzles,
                start_time.elapsed().as_secs_f32(),
                (start_time.elapsed().as_secs_f32() / n as f32)
                    * (num_root_puzzles as f32 - n as f32),
                puzzle_root
            );

            let puzzle_evaluation = evaluate_puzzle_candidate(processed_candidate.clone());

            match puzzle_evaluation {
                PuzzleCandidateEvaluation::Approve(_) => {
                    num_approved.fetch_add(1, Ordering::Relaxed);
                }
                PuzzleCandidateEvaluation::Deny => {
                    num_denied.fetch_add(1, Ordering::Relaxed);
                }
                PuzzleCandidateEvaluation::ManualReview(_) => {
                    num_manual_review.fetch_add(1, Ordering::Relaxed);
                }
            }

            for processed_candidate in processed_candidate.solutions {
                print!(
                    "Goes to road: {}, solution: {}",
                    processed_candidate.goes_to_road,
                    processed_candidate.moves
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                if processed_candidate.moves.len() % 2 == 0 {
                    print!(" *");
                }
                if !processed_candidate.pure_recaptures_end_sequence.is_empty() {
                    print!(" (desperado defense: {})", processed_candidate.pure_recaptures_end_sequence.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" "));
                }
                if let Some(skipped_line) = &processed_candidate.trivial_desperado_defense_skipped {
                    print!(" (trivial desperado defense skipped: {})", skipped_line.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" "));
                }
                if puzzle_evaluation == PuzzleCandidateEvaluation::Approve(processed_candidate.clone()) {
                    print!(" (approved solution)");
                } else if puzzle_evaluation == PuzzleCandidateEvaluation::Deny {
                    print!(" (denied solution)");
                }
                else if puzzle_evaluation == PuzzleCandidateEvaluation::ManualReview(processed_candidate.clone()) {
                    print!(" (primary candidate solution)");
                }
                println!();
            }
            println!();

            if n % 10 == 0 {
                println!(
                    "{}/{} puzzles go to a road, {}/{} goes to road with single solution, {}/{} puzzles have a single solution, {}/{} puzzles only have a 1-long solution and no desperado defense",
                    num_goes_to_road.load(Ordering::Relaxed),
                    n,
                    num_single_solution_to_road.load(Ordering::Relaxed),
                    n,
                    num_single_solution.load(Ordering::Relaxed),
                    n,
                    num_one_move_solution_and_no_desperado.load(Ordering::Relaxed),
                    n,
                );
                println!("{} puzzles approved, {} puzzles denied, {} puzzles need manual review",
                    num_approved.load(Ordering::Relaxed),
                    num_denied.load(Ordering::Relaxed),
                    num_manual_review.load(Ordering::Relaxed),
                );
                println!("Time usage stats:");
                println!("{}", stats);
            }
            tinue_candidate
        })
        .collect();
    puzzle_candidates
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PuzzleCandidateEvaluation<const S: usize> {
    Approve(TinueLineCandidate<S>),
    Deny,
    ManualReview(TinueLineCandidate<S>), // Contains the primary candidate solution
}

pub fn evaluate_candidate_line<const S: usize>(
    solution: TinueLineCandidate<S>,
) -> PuzzleCandidateEvaluation<S> {
    use PuzzleCandidateEvaluation::*;

    if solution.goes_to_road {
        Approve(solution)
    } else if solution.moves.len() == 1 {
        if solution.pure_recaptures_end_sequence.is_empty() {
            // A single move solution that does not go to a road nor a clear tinue is not a good candidate
            Deny
        } else {
            // Manually review whether the single-move puzzle is interesting enough
            ManualReview(solution)
        }
    } else if !solution.pure_recaptures_end_sequence.is_empty() {
        // Longer puzzles that end in a clear tinue are good candidates
        Approve(solution)
    } else {
        ManualReview(solution)
    }
}

pub fn evaluate_puzzle_candidate<const S: usize>(
    candidate: TinuePuzzleCandidate2<S>,
) -> PuzzleCandidateEvaluation<S> {
    use PuzzleCandidateEvaluation::*;
    if candidate.solutions.len() == 1 {
        return evaluate_candidate_line(candidate.solutions[0].clone());
    }
    let any_solution_goes_to_road = candidate
        .solutions
        .iter()
        .any(|solution| solution.goes_to_road);

    let (longest_solution, longest_line_length) = candidate
        .solutions
        .iter()
        .map(|solution| (solution, solution.num_moves()))
        .max_by_key(|(_, len)| *len)
        .unwrap();

    // If the longest line is a road, strictly longer than the others, and at least 2 moves longer than any non-road line,
    // it's clearly better than the other lines
    if longest_solution.goes_to_road {
        // If the longest solution goes to a road, but a non-road alternative exists that is only one move shorter,
        // send to manual review always
        if let Some(long_non_road_solution) = candidate
            .solutions
            .iter()
            .filter(|solution| {
                !solution.goes_to_road && (solution.num_moves() + 1) >= longest_solution.num_moves()
            })
            .max_by_key(|solution| solution.num_moves())
        {
            return ManualReview(long_non_road_solution.clone());
        }

        // Otherwise, if it's strictly longer than all other lines, approve
        if candidate.solutions.iter().all(|solution| {
            solution == longest_solution || solution.num_moves() < longest_solution.num_moves()
        }) {
            return Approve(longest_solution.clone());
        }

        // If there are multiple longest solutions that go to a road,
        // find the one with the most blocking defensive placements
        let longest_solutions = candidate
            .solutions
            .iter()
            .filter(|solution| solution.goes_to_road && solution.num_moves() == longest_line_length)
            .collect::<Vec<_>>();
        assert!(longest_solutions.len() >= 2);
        if let Some(most_blocking_placements) = longest_solutions.iter().find(|solution| {
            longest_solutions.iter().all(|other_solution| {
                *solution == other_solution
                    || solution.num_blocking_defensive_placements()
                        > other_solution.num_blocking_defensive_placements()
            })
        }) {
            return Approve((*most_blocking_placements).clone());
        }
    }

    // If no road lines exist, but the longest line is strictly longer than the others,
    // this line is clearly better
    // Approve it as long as the line is a good puzzle by itself
    if !any_solution_goes_to_road
        && candidate.solutions.iter().all(|solution| {
            solution == longest_solution || solution.num_moves() < longest_solution.num_moves()
        })
    {
        return evaluate_candidate_line(longest_solution.clone());
    }
    ManualReview(longest_solution.clone())
}

#[derive(Clone)]
pub struct TinuePuzzleCandidate<const S: usize> {
    pub position: Position<S>,
    pub solutions: Vec<(Vec<Move<S>>, bool)>,
}

impl<const S: usize> From<TinuePuzzleCandidate2<S>> for TinuePuzzleCandidate<S> {
    fn from(candidate: TinuePuzzleCandidate2<S>) -> Self {
        TinuePuzzleCandidate {
            position: candidate.position,
            solutions: candidate
                .solutions
                .into_iter()
                .map(|solution| (solution.moves, solution.goes_to_road))
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct TinuePuzzleCandidate2<const S: usize> {
    pub position: Position<S>,
    pub solutions: Vec<TinueLineCandidate<S>>,
}

impl<const S: usize> TinuePuzzleCandidate2<S> {
    pub fn reanalyze(self) -> Self {
        process_full_tinue(TinuePuzzleCandidate::from(self))
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct TinueLineCandidateRow {
    pub game_id: u64,
    pub tps: String,
    pub solution: String,
    pub goes_to_road: bool,
    pub pure_recaptures_end_sequence: String,
    pub trivial_desperado_defense_skipped: Option<String>,
}

impl TinueLineCandidateRow {
    fn from_puzzle_candidate<const S: usize>(
        candidate: TinuePuzzleCandidate2<S>,
        game_id: u64,
        root_tps: &str,
    ) -> Vec<Self> {
        candidate
            .solutions
            .into_iter()
            .map(|solution| TinueLineCandidateRow {
                game_id,
                tps: root_tps.to_string(),
                solution: solution
                    .moves
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                goes_to_road: solution.goes_to_road,
                pure_recaptures_end_sequence: solution
                    .pure_recaptures_end_sequence
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                trivial_desperado_defense_skipped: solution
                    .trivial_desperado_defense_skipped
                    .as_ref()
                    .map(|v| {
                        v.iter()
                            .map(|m| m.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    }),
            })
            .collect()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TinueLineCandidate<const S: usize> {
    pub moves: Vec<Move<S>>,
    pub goes_to_road: bool,
    // If the line doesn't end in a road,
    // but ends in a position where the attack only has to make pure recaptures until a forced road,
    // this additional line is stored here
    // It is not part of the puzzle, because each attacking move may have more than one solution
    pub pure_recaptures_end_sequence: Vec<Move<S>>,
    // The last defending move has been modified
    // A better move was technically available, but it's refuted by a trivial recapture,
    // and also refuted by other moves as well, so that the puzzle could not be extended to a road with that move
    // Because of this, the last defending move has been replaced by an immediately losing move
    pub trivial_desperado_defense_skipped: Option<Vec<Move<S>>>,
}

impl<const S: usize> TinueLineCandidate<S> {
    pub fn num_moves(&self) -> usize {
        self.moves.len() / 2
    }

    /// Number of walls/caps placed by the defender in this line
    pub fn num_blocking_defensive_placements(&self) -> usize {
        self.moves
            .iter()
            .enumerate()
            .filter(|(i, mv)| {
                i % 2 == 1 && matches!(mv.expand(), ExpMove::Place(Role::Cap | Role::Wall, _))
            })
            .count()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct DesperadoDefenseLine<const S: usize> {
    pub moves: Vec<Move<S>>,
    pub only_trivial_recaptures: bool,
}

impl<const S: usize> Ord for DesperadoDefenseLine<S> {
    // A higher ordering means a better line for the defender
    // Prefer lines with non-trivial recaptures, then prefer longer lines
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.only_trivial_recaptures
            .cmp(&other.only_trivial_recaptures)
            .reverse()
            .then(self.moves.len().cmp(&other.moves.len()))
    }
}

impl<const S: usize> PartialOrd for DesperadoDefenseLine<S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Position is a known loss for the side-to-move
pub fn find_desperado_defense_lines<const S: usize>(
    position: &mut Position<S>,
) -> Option<DesperadoDefenseLine<S>> {
    if let Some((last_move, unique_win)) = find_last_defending_move(position) {
        let mut moves = vec![last_move];
        if let Some(unique_move) = unique_win {
            moves.push(unique_move);
        }
        return Some(DesperadoDefenseLine {
            moves,
            only_trivial_recaptures: true,
        });
    }

    let mut defender_moves = vec![];
    position.generate_moves(&mut defender_moves);

    defender_moves.retain(
        // Only keep defending moves that don't lead to an immediate win
        |mv| {
            assert!(position.move_is_legal(*mv));
            let reverse_move = position.do_move(*mv);
            let mut child_moves = vec![];
            position.generate_moves(&mut child_moves);
            for child_move in child_moves {
                let reverse_child_move = position.do_move(child_move);
                if position.game_result() == Some(GameResult::win_by(!position.side_to_move())) {
                    position.reverse_move(reverse_child_move);
                    position.reverse_move(reverse_move);
                    return false; // This move leads to an immediate win
                }
                position.reverse_move(reverse_child_move);
            }
            position.reverse_move(reverse_move);
            true // This move does not lead to an immediate win
        },
    );

    assert!(!defender_moves.is_empty());

    let mut best_defense: Option<DesperadoDefenseLine<S>> = None;

    'defender_loop: for defender_move in defender_moves {
        let ExpMove::Move(origin_square, direction, stack_movement) = defender_move.expand() else {
            return None; // Non-losing placements are always a good enough defense
        };
        assert!(position.move_is_legal(defender_move));
        let mut destination_square = origin_square;

        let mut our_squares_gained = vec![];
        let mut squares_affected = vec![];

        for top_piece in position.top_stones_left_behind_by_move(origin_square, &stack_movement) {
            if destination_square != defender_move.destination_square()
                && top_piece.is_some_and(|piece| piece.color() == position.side_to_move())
                && position.top_stones()[destination_square]
                    .is_some_and(|piece| piece.color() != position.side_to_move())
            {
                // If their move captured our piece on any square except the destination square,
                // it's not a desperado defense
                return None;
            }
            if top_piece.is_some_and(|piece| piece.color() != position.side_to_move())
                && position.top_stones()[destination_square]
                    .is_none_or(|piece| piece.color() == position.side_to_move())
            {
                our_squares_gained.push(destination_square);
            }
            squares_affected.push(destination_square);
            destination_square = destination_square
                .go_direction(direction)
                .unwrap_or(destination_square);
        }

        let reverse_move = position.do_move(defender_move);

        let mut attacker_moves = vec![];
        position.generate_moves(&mut attacker_moves);
        let mut attacker_moves: Vec<(Move<S>, bool)> = attacker_moves
            .into_iter()
            .filter_map(
                // Only keep trivial recapture moves, either pure captures,
                // or captures using a flat left behind by the previous defending move
                |mv| {
                    let ExpMove::Move(origin_square, _, stack_movement) = mv.expand() else {
                        return None; // Placements are never trivial recaptures
                    };
                    // If the move is a pure recapture, we can use it
                    let is_pure_spread = position
                        .top_stones_left_behind_by_move(origin_square, &stack_movement)
                        .all(|piece| {
                            piece.is_some_and(|piece| {
                                piece.color() == position.side_to_move() && piece.is_road_piece()
                            })
                        });
                    let is_almost_pure_spread = position
                        .top_stones_left_behind_by_move(origin_square, &stack_movement)
                        .eq([
                            None,
                            Some(Piece::from_role_color(Role::Flat, position.side_to_move())),
                        ]);
                    let just_got_square = our_squares_gained.contains(&origin_square);
                    if mv.destination_square() == defender_move.destination_square() {
                        if is_almost_pure_spread && just_got_square
                            || is_pure_spread && squares_affected.contains(&mv.origin_square())
                        {
                            // If the move is a two-long spread using a flat left behind by the previous defending move,
                            // it's an even more trivial recapture
                            return Some((mv, true));
                        } else if is_pure_spread {
                            // Any pure spreads are less trivial, but we can still use them
                            return Some((mv, false));
                        }
                    }
                    // Not a trivial recapture
                    None
                },
            )
            .collect();

        // We want to try the most trivial recaptures first
        attacker_moves.sort_by_key(|(_, is_extra_trivial)| *is_extra_trivial);
        attacker_moves.reverse();

        for (attacker_move, is_extra_trivial) in attacker_moves {
            let reverse_attacker_move = position.do_move(attacker_move);
            let results = find_desperado_defense_lines(position);
            position.reverse_move(reverse_attacker_move);

            // This always chooses the first attacking line as the best. TODO: Choose smarter
            if let Some(mut result) = results {
                result.moves.insert(0, attacker_move);
                result.moves.insert(0, defender_move);
                result.only_trivial_recaptures = result.only_trivial_recaptures && is_extra_trivial;

                if let Some(best_result) = best_defense.as_mut() {
                    if result > *best_result {
                        *best_result = result;
                    }
                } else {
                    best_defense = Some(result);
                }
                position.reverse_move(reverse_move);
                continue 'defender_loop; // We found a winning followup, so we can skip the rest of the attacker moves
            }
        }
        position.reverse_move(reverse_move);
        // If we reach here, it means that the attacker has no trivial recaptures against this defense,
        // which in this context means we found a good enough defense.
        return None;
    }
    best_defense
}

pub fn process_full_tinue<const S: usize>(
    puzzle: TinuePuzzleCandidate<S>,
) -> TinuePuzzleCandidate2<S> {
    let processed_candidates: Vec<_> = puzzle
        .solutions
        .into_iter()
        .map(|(mut solution, goes_to_road)| {
            let mut position: Position<S> = puzzle.position.clone();
            for mv in solution.iter() {
                assert!(
                    position.move_is_legal(*mv),
                    "Move {} is not legal in position {}, root position {}, solution {}",
                    mv,
                    position.to_fen(),
                    puzzle.position.to_fen(),
                    solution
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                position.do_move(*mv);
            }
            if goes_to_road {
                return TinueLineCandidate {
                    moves: solution.clone(),
                    goes_to_road: true,
                    pure_recaptures_end_sequence: vec![],
                    trivial_desperado_defense_skipped: None,
                };
            }
            if let Some(desperado_result) = find_desperado_defense_lines(&mut position) {
                if !desperado_result.only_trivial_recaptures {
                    // If the defense is not super trivial, don't skip it
                    TinueLineCandidate {
                        moves: solution.clone(),
                        goes_to_road: false,
                        pure_recaptures_end_sequence: desperado_result.moves,
                        trivial_desperado_defense_skipped: None,
                    }
                } else {
                    // If the defense is trivial, generate a new immediate losing move at the end
                    // and skip the full defense
                    let mut defending_moves = vec![];
                    position.generate_moves(&mut defending_moves);
                    // Only retain defending moves that lead to an immediate loss
                    defending_moves.retain(|mv| {
                        let reverse_move = position.do_move(*mv);
                        let mut child_moves = vec![];
                        position.generate_moves(&mut child_moves);
                        for child_move in child_moves {
                            let reverse_child_move = position.do_move(child_move);
                            if position.game_result()
                                == Some(GameResult::win_by(!position.side_to_move()))
                            {
                                position.reverse_move(reverse_child_move);
                                position.reverse_move(reverse_move);
                                return true; // This move leads to an immediate win
                            }
                            position.reverse_move(reverse_child_move);
                        }
                        position.reverse_move(reverse_move);
                        false // This move does not lead to an immediate win
                    });

                    let (defender_move, attacker_move) =
                        find_last_defending_move_among_moves(&mut position, &defending_moves)
                            .unwrap();

                    solution.push(defender_move);
                    if let Some(attacker_move) = attacker_move {
                        solution.push(attacker_move);
                    }

                    TinueLineCandidate {
                        moves: solution,
                        goes_to_road: true,
                        pure_recaptures_end_sequence: vec![],
                        trivial_desperado_defense_skipped: Some(desperado_result.moves),
                    }
                }
            } else {
                TinueLineCandidate {
                    moves: solution.clone(),
                    goes_to_road: false,
                    pure_recaptures_end_sequence: vec![],
                    trivial_desperado_defense_skipped: None,
                }
            }
        })
        .collect();
    TinuePuzzleCandidate2 {
        position: puzzle.position,
        solutions: processed_candidates,
    }
}

pub fn extract_possible_full_tinues<const S: usize>(
    mut position: Position<S>,
    first_move: Move<S>,
    stats: &Stats,
) -> TinuePuzzleCandidate<S> {
    assert!(position.move_is_legal(first_move));
    position.do_move(first_move);
    let mut moves = vec![first_move];
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
