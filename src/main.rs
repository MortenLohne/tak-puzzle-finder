use std::collections::BTreeSet;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use board_game_traits::{Color, GameResult, Position as PositionTrait};
use chrono::{DateTime, NaiveDateTime, Utc};
use pgn_traits::PgnPosition;
use rusqlite::{Connection, ErrorCode};
use tiltak::position::{self, Komi, Move, Position};
use tiltak::search;
use topaz_tak::board::{Board5, Board6};

const TILTAK_SHALLOW_NODES: u32 = 50_000;
const TILTAK_DEEP_NODES: u32 = 2_000_000;

const TOPAZ_FIRST_MOVE_NODES: usize = 10_000_000;
const TOPAZ_SECOND_MOVE_NODES: usize = 20_000_000;
const TOPAZ_AVOIDANCE_NODES: usize = 5_000_000;

fn main() {
    // Create a connection pool
    //  for MySQL, use MySqlPoolOptions::new()
    //  for SQLite, use SqlitePoolOptions::new()
    //  etc.
    let mut db_conn = Connection::open("games_anon.db").unwrap();

    let games = read_non_bot_games(&mut db_conn).unwrap();

    let mut relevant_games: Vec<PlaytakGame> = games
        .into_iter()
        .filter(|game| {
            game.size == 5
                && game.moves.len() > 4
                && !game.is_bot_game()
                && game.game_is_legal()
                && game.has_standard_piece_count()
        })
        .collect();

    relevant_games.shuffle(&mut thread_rng());
    let num_games = relevant_games.len();

    let start_time = Instant::now();
    let stats = Arc::new(Stats::default());
    let games_processed = Arc::new(AtomicU64::new(0));

    let puzzles_pool = Connection::open("puzzles.db").unwrap();

    puzzles_pool
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

    let mut stmt = puzzles_pool.prepare("INSERT OR IGNORE INTO games (
            id, date, size, player_white, player_black, notation, result, timertime, timerinc, rating_white, rating_black, unrated, tournament, komi, has_been_analyzed)
            values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 0)").unwrap();

    println!(
        "Copying {} games from the input database. This may take a few minutes.",
        num_games
    );
    for game in relevant_games.iter() {
        stmt.execute(rusqlite::params![
            game.id,
            game.date_time,
            game.size,
            game.player_white,
            game.player_black,
            game.moves
                .iter()
                .fold(String::with_capacity(200), |mut moves, mv| {
                    match game.size {
                        5 => write!(moves, "{} ", mv.to_string::<5>()).unwrap(),
                        6 => write!(moves, "{} ", mv.to_string::<6>()).unwrap(),
                        _ => unimplemented!(),
                    };
                    moves
                })
                .trim(),
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

    puzzles_pool
        .execute(
            "CREATE TABLE IF NOT EXISTS puzzles (
            game_id	INTEGER NOT NULL,
            tps	TEXT PRIMARY KEY,
            solution TEXT NOT NULL,
            tiltak_eval	REAL NOT NULL,
            tiltak_second_move_eval	REAL NOT NULL,
            tinue_length INTEGER,
            tinue_avoidance_length INTEGER,
            tiltak_pv_length INTEGER NOT NULL,
            tiltak_second_pv_length INTEGER NOT NULL,
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

    relevant_games.par_iter().for_each_init(
        || Connection::open("puzzles.db").unwrap(),
        |puzzle_conn, game| {
            for possible_puzzle in generate_possible_puzzle::<5>(&stats, game) {
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
                } else if let Some(puzzle) = possible_puzzle.make_real_puzzle::<5>() {
                    store_puzzle(puzzle_conn, puzzle);
                }
            }

            set_game_analyzed(puzzle_conn, game.id);

            if games_processed.fetch_add(1, Ordering::SeqCst) % 200 == 0 {
                println!(
                    "Checked {}/{} games in {}s",
                    games_processed.load(Ordering::SeqCst),
                    num_games,
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
        "INSERT INTO failed_puzzles (game_id, tps, failure_kind) values (?1, ?2, ?3)",
        rusqlite::params![game_id, tps, failure_kind],
    )
    .unwrap();
}

fn store_topaz_missed_tinues(conn: &mut Connection, game_id: u32, tps: &str) {
    conn.execute(
        "INSERT INTO topaz_missed_tinues (game_id, tps) values (?1, ?2)",
        rusqlite::params![game_id, tps],
    )
    .unwrap();
}

fn store_puzzle(puzzles_pool: &mut Connection, puzzle: Puzzle) {
    let Puzzle {
        game_id,
        tps,
        solution,
        tiltak_eval,
        tiltak_second_move_eval,
        tiltak_pv_length,
        tiltak_second_pv_length,
        tinue_length,
        tinue_avoidance_length,
    } = puzzle;

    while let Err(rusqlite::Error::SqliteFailure(err, _)) = puzzles_pool.execute("INSERT INTO puzzles (game_id, tps, solution, tiltak_eval, tiltak_second_move_eval, tinue_length, tinue_avoidance_length, tiltak_pv_length, tiltak_second_pv_length) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)", rusqlite::params![
        game_id as u32,
        tps.clone(),
        solution.clone(),
        tiltak_eval,
        tiltak_second_move_eval,
        tinue_length,
        tinue_avoidance_length,
        tiltak_pv_length,
        tiltak_second_pv_length,
    ]) {
        if err.code == ErrorCode::ConstraintViolation {
            println!(
                "Failed to insert \"{}\" from game ${} into DB due to uniqueness constraint: {}",
                tps, game_id, err
            );
            break;
        }
        println!(
            "Failed to insert \"{}\" from game ${} into DB. Retrying in 1s: {}",
            tps, game_id, err
        );
        thread::sleep(Duration::from_secs(1));
    }
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
                NaiveDateTime::from_timestamp_opt(row.get::<_, i64>(1).unwrap() / 1000, 0).unwrap(),
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
) -> impl Iterator<Item = PossiblePuzzle> + 'a {
    let mut position = <Position<S>>::start_position();
    let mut last_move_was_tinue = false;

    game.moves.iter().filter_map(move |mv| {
        let last_tps = position.to_fen();

        position.do_move(mv.clone());
        if position.game_result().is_some() {
            return None;
        }
        // Tinue is only possible after white has played S - 2 moves
        if position.half_moves_played() <= (S - 2) * 2 {
            return None;
        }

        let tps = position.to_fen();

        let tiltak_analysis_shallow = {
            let start_time = Instant::now();
            let result = tiltak_search(position.clone(), TILTAK_SHALLOW_NODES);

            stats.tiltak_non_tinue_short.record(start_time.elapsed());

            result
        };

        let mut immediate_wins = vec![];
        position.generate_moves(&mut immediate_wins);
        immediate_wins.retain(|mv| {
            let reverse_move = position.do_move(mv.clone());
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
            previous_move: mv.clone(),
            immediate_wins,
            followup_move: None,
            topaz_tinue: topaz_result.clone(),
            topaz_tinue_avoidance: None,
            tiltak_analysis_shallow: tiltak_analysis_shallow.clone(),
            tiltak_analysis_deep: None,
            last_move_was_tinue,
        };

        last_move_was_tinue = false;

        // If we have at least one half-decent move, check for tinue avoidance puzzle
        if tiltak_analysis_shallow.score_first > 0.1
            && !matches!(
                topaz_result,
                TopazResult::RoadWin | TopazResult::NonUniqueTinue(_)
            )
        {
            let start_time = Instant::now();
            let tinue_avoidance = topaz_tinue_avoidance(&mut position, &tiltak_analysis_shallow);
            stats.topaz_tinue_avoidance.record(start_time.elapsed());

            possible_puzzle.topaz_tinue_avoidance = Some(tinue_avoidance);
        };

        match topaz_result {
            TopazResult::NoTinue | TopazResult::AbortedFirst | TopazResult::AbortedSecond(_) => {
                if matches!(
                    possible_puzzle.topaz_tinue_avoidance,
                    Some(TopazAvoidanceResult::Defense(_))
                ) || tiltak_analysis_shallow.score_first > 0.4
                    && tiltak_analysis_shallow.score_second < 0.6
                    && tiltak_analysis_shallow.score_first - tiltak_analysis_shallow.score_second
                        > 0.3
                {
                    let tiltak_start_time = Instant::now();
                    let tiltak_analysis_deep = tiltak_search(position.clone(), TILTAK_DEEP_NODES);

                    possible_puzzle.tiltak_analysis_deep = Some(tiltak_analysis_deep);

                    stats
                        .tiltak_non_tinue_long
                        .record(tiltak_start_time.elapsed());
                }
            }
            TopazResult::RoadWin | TopazResult::NonUniqueTinue(_) => last_move_was_tinue = true,
            TopazResult::Tinue(_) => {
                let tiltak_start_time = Instant::now();
                let tiltak_analysis_deep = tiltak_search(position.clone(), TILTAK_DEEP_NODES);

                possible_puzzle.tiltak_analysis_deep = Some(tiltak_analysis_deep);

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

#[derive(Debug, Clone)]
struct Puzzle {
    game_id: u64,
    tps: String,
    solution: String,
    tiltak_eval: f32,
    tiltak_second_move_eval: f32,
    tiltak_pv_length: u32,
    tiltak_second_pv_length: u32,
    tinue_length: Option<u32>,
    tinue_avoidance_length: Option<u32>,
}

struct PossiblePuzzle {
    playtak_game: PlaytakGame,
    tps: String,
    previous_tps: String,
    previous_move: Move,
    immediate_wins: Vec<Move>,
    followup_move: Option<Move>,
    topaz_tinue: TopazResult,
    topaz_tinue_avoidance: Option<TopazAvoidanceResult>,
    tiltak_analysis_shallow: TiltakResult,
    tiltak_analysis_deep: Option<TiltakResult>,
    last_move_was_tinue: bool,
}

impl PossiblePuzzle {
    fn make_real_puzzle<const S: usize>(&self) -> Option<Puzzle> {
        let tiltak_eval = self.tiltak_analysis_deep.as_ref()?;
        let mut puzzle = Puzzle {
            game_id: self.playtak_game.id,
            tps: self.tps.clone(),
            solution: tiltak_eval
                .pv_first
                .first()
                .map(|mv| mv.to_string::<S>())
                .unwrap_or_default(),
            tiltak_eval: tiltak_eval.score_first,
            tiltak_second_move_eval: tiltak_eval.score_second,
            tiltak_pv_length: tiltak_eval.pv_first.len() as u32,
            tiltak_second_pv_length: tiltak_eval.pv_second.len() as u32,
            tinue_length: None,
            tinue_avoidance_length: None,
        };
        if let TopazResult::Tinue(moves) = &self.topaz_tinue {
            puzzle.tinue_length = Some(moves.len() as u32 + 1);
            puzzle.solution = moves[0].to_string::<S>();
        }
        if let Some(TopazAvoidanceResult::Defense(tinue_avoidance)) = &self.topaz_tinue_avoidance {
            puzzle.tinue_avoidance_length = Some(tinue_avoidance.longest_refutation_length + 2);
            puzzle.solution = tinue_avoidance.defense.to_string::<S>();
        }
        Some(puzzle)
    }
}

#[derive(Clone)]
struct TinueAvoidance {
    defense: Move,
    longest_refutation_length: u32,
}

#[derive(Default)]
struct Stats {
    topaz_tinue_first: TimeTracker,
    topaz_tinue_second: TimeTracker,
    topaz_tinue_avoidance: TimeTracker,
    tiltak_tinue: TimeTracker,
    tiltak_non_tinue_short: TimeTracker,
    tiltak_non_tinue_long: TimeTracker,
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
        println!("{} events in {}s total time, average {:.2}s, longest {:.2}s, top 50% {:.2}s, top 1% {:.2}s, top 0.1% {:.2}s", 
            times.len(),
            total_time.as_secs(),
            total_time.as_secs_f32() / times.len() as f32,
            times.last().cloned().unwrap_or_default().as_secs_f32(),
            times.get(times.len() / 2).cloned().unwrap_or_default().as_secs_f32(),
            times.get(990 * times.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
            times.get(999 * times.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
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
struct TiltakResult {
    score_first: f32,
    pv_first: Vec<Move>,
    score_second: f32,
    pv_second: Vec<Move>,
}

fn tiltak_search<const S: usize>(position: Position<S>, nodes: u32) -> TiltakResult {
    let settings1 = search::MctsSetting::default().arena_size_for_nodes(nodes);
    let mut tree1 = search::MonteCarloTree::with_settings(position.clone(), settings1);
    for _ in 0..nodes {
        match tree1.select() {
            Some(_) => (),
            None => {
                eprintln!("Tiltak search aborted early due to oom");
                break;
            }
        }
    }
    let (best_move, score) = tree1.best_move();

    let settings2 = search::MctsSetting::default()
        .arena_size_for_nodes(nodes)
        .exclude_moves(vec![best_move]);
    let mut tree2 = search::MonteCarloTree::with_settings(position, settings2);
    for _ in 0..nodes {
        match tree2.select() {
            Some(_) => (),
            None => {
                eprintln!("Tiltak search aborted early due to oom");
                break;
            }
        }
    }

    // It's possible that the second move actually scores better than the first move
    // In that case, swap the moves, to make later processing easier
    if tree2.best_move().1 > score {
        TiltakResult {
            score_first: tree2.best_move().1,
            pv_first: tree2.pv().collect(),
            score_second: score,
            pv_second: tree1.pv().collect(),
        }
    } else {
        TiltakResult {
            score_first: score,
            pv_first: tree1.pv().collect(),
            score_second: tree2.best_move().1,
            pv_second: tree2.pv().collect(),
        }
    }
}

#[derive(Clone)]
enum TopazResult {
    NoTinue,
    RoadWin,
    Tinue(Vec<Move>),
    AbortedFirst,
    AbortedSecond(Vec<Move>),
    NonUniqueTinue(Vec<Move>),
}

fn topaz_search<const S: usize>(tps: &str, stats: &Stats) -> TopazResult {
    match S {
        5 => topaz_search_5s(tps, stats),
        6 => topaz_search_6s(tps, stats),
        _ => unimplemented!(),
    }
}

fn topaz_search_5s(tps: &str, stats: &Stats) -> TopazResult {
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
                .map(|mv| {
                    Move::from_string::<5>(mv.to_ptn::<Board5>().trim_end_matches('*')).unwrap()
                })
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

fn topaz_search_6s(tps: &str, stats: &Stats) -> TopazResult {
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
            .map(|mv| Move::from_string::<6>(mv.to_ptn::<Board6>().trim_end_matches('*')).unwrap())
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
enum TopazAvoidanceResult {
    Aborted,
    MultipleDefenses,
    NoDefense,
    Defense(TinueAvoidance),
}

fn topaz_tinue_avoidance<const S: usize>(
    position: &mut Position<S>,
    shallow_tiltak_analysis: &TiltakResult,
) -> TopazAvoidanceResult {
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
        let reverse_move = position.do_move(mv.clone());
        let board: Board5 = topaz_tak::board::Board5::try_from_tps(&position.to_fen()).unwrap();
        position.reverse_move(reverse_move);
        let mut tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
            .quiet()
            .limit(TOPAZ_AVOIDANCE_NODES);
        match tinue_search.is_tinue() {
            None => has_aborted = false, // If search aborts in one child, we can still conclude `MultipleDefenses` if two other children are not tinue
            Some(false) if defense.is_some() => return TopazAvoidanceResult::MultipleDefenses,
            Some(false) => defense = Some(mv.clone()),
            Some(true) => {
                if let Some(response) = tinue_search
                    .principal_variation()
                    .get(0)
                    .and_then(|mv| Move::from_string::<S>(&mv.to_ptn::<Board5>()).ok())
                {
                    refutations.push([mv.clone(), response]);
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
            defense: mv,
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
    moves: Vec<Move>,
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
        fn game_is_legal_sized<const S: usize>(moves: &[Move], komi: Komi) -> bool {
            let mut position: Position<S> = Position::start_position_with_komi(komi);
            let mut legal_moves = vec![];
            for mv in moves {
                if position.game_result().is_some() {
                    return false;
                }
                position.generate_moves(&mut legal_moves);
                if !legal_moves.contains(mv) {
                    return false;
                }
                legal_moves.clear();
                position.do_move(mv.clone());
            }
            true
        }
        match self.size {
            3 => game_is_legal_sized::<3>(&self.moves, self.komi),
            4 => game_is_legal_sized::<4>(&self.moves, self.komi),
            5 => game_is_legal_sized::<5>(&self.moves, self.komi),
            6 => game_is_legal_sized::<6>(&self.moves, self.komi),
            7 => game_is_legal_sized::<7>(&self.moves, self.komi),
            8 => game_is_legal_sized::<8>(&self.moves, self.komi),
            _ => unreachable!(),
        }
    }
}

impl TryFrom<GameRow> for PlaytakGame {
    type Error = ();
    fn try_from(row: GameRow) -> Result<Self, ()> {
        let moves = match row.size {
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
            size: row.size.try_into().map_err(|_| ())?,
            player_white: row.player_white,
            player_black: row.player_black,
            moves,
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

fn parse_notation<const S: usize>(notation: &str) -> Vec<Move> {
    if notation.is_empty() {
        vec![]
    } else {
        notation
            .split(',')
            .map(Move::from_string_playtak::<S>)
            .collect()
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
