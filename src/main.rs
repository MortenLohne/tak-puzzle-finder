use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;
use std::{sync::Mutex, time::Duration};

use rayon::prelude::*;
use sqlx::sqlite::SqliteConnectOptions;
use tokio::task;

use board_game_traits::{Color, GameResult, Position as PositionTrait};
use chrono::{DateTime, NaiveDateTime, Utc};
use pgn_traits::PgnPosition;
use sqlx::{Connection, SqliteConnection, SqlitePool};
use tiltak::position::{self, Komi, Move, Position};
use tiltak::search;
use topaz_tak::board::{Board5, Board6};

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    // Create a connection pool
    //  for MySQL, use MySqlPoolOptions::new()
    //  for SQLite, use SqlitePoolOptions::new()
    //  etc.
    let mut db_conn = SqliteConnection::connect("sqlite:games_anon.db").await?;

    match sqlx::query("ALTER TABLE games ADD has_been_analyzed INTEGER DEFAULT 0")
        .execute(&mut db_conn)
        .await
    {
        Ok(_) => (),
        Err(sqlx::Error::Database(err)) if err.message().starts_with("duplicate column name") => (),
        Err(err) => panic!("{}", err),
    }

    let games = read_non_bot_games(&mut db_conn).await.unwrap();

    let start_time = Instant::now();
    let stats = Stats::default();
    let games_processed = AtomicU64::new(0);

    let puzzles_pool = SqlitePool::connect_with(
        SqliteConnectOptions::new()
            .filename("puzzles.db")
            .create_if_missing(true),
    )
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS \"puzzles\" (
        \"game_id\"	INTEGER NOT NULL,
        \"tps\"	TEXT NOT NULL UNIQUE,
        \"solution\" TEXT NOT NULL,
        \"tiltak_eval\"	REAL NOT NULL,
        \"tiltak_second_move_eval\"	REAL NOT NULL,
        \"tinue_length\" INTEGER,
        \"tinue_avoidance_length\" INTEGER,
        PRIMARY KEY(\"tps\")
    )",
    )
    .execute(&puzzles_pool)
    .await
    .unwrap();

    let (puzzle_sender, puzzle_receiver) = mpsc::channel();
    let (game_sender, game_receiver) = mpsc::channel();

    task::spawn_blocking(move || {
        games
            .par_iter()
            .filter(|game| {
                game.size == 5
                    && !game.is_bot_game()
                    && game.game_is_legal()
                    && game.has_standard_piececount()
                    && game.player_white.is_some()
                    && game.player_black.is_some()
            })
            .for_each(|game| {
                let mut has_been_aborted = false;
                for possible_puzzle in generate_possible_puzzle::<5>(&stats, game) {
                    if matches!(
                        possible_puzzle.topaz_tinue,
                        TopazResult::AbortedFirst | TopazResult::AbortedSecond(_)
                    ) || matches!(
                        possible_puzzle.topaz_tinue_avoidance,
                        Some(TopazAvoidanceResult::Aborted)
                    ) {
                        has_been_aborted = true;
                    }
                    if let Some(puzzle) = possible_puzzle.make_real_puzzle::<5>() {
                        puzzle_sender.send(puzzle).unwrap();
                    }
                }

                game_sender.send((game.id, has_been_aborted)).unwrap();

                if games_processed.fetch_add(1, Ordering::SeqCst) % 50 == 0 {
                    println!(
                        "Checked {}/{} games in {}s",
                        games_processed.load(Ordering::SeqCst),
                        games.len(),
                        start_time.elapsed().as_secs()
                    );
                    print!("Topaz tinue: ");
                    stats.topaz_tinue.print_full();
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
            })
    });

    task::spawn(async move {
        loop {
            let puzzle = puzzle_receiver.recv().unwrap();
            let Puzzle {
                game_id,
                tps,
                solution,
                tiltak_eval,
                tiltak_second_move_eval,
                tinue_length,
                tinue_avoidance_length,
            } = puzzle;

            while let Err(err) =
            sqlx::query(
            "INSERT INTO puzzles (game_id, tps, solution, tiltak_eval, tiltak_second_move_eval, tinue_length, tinue_avoidance_length) values (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
        ).bind(game_id as u32)
        .bind(tps.clone())
        .bind(solution.clone())
        .bind(tiltak_eval)
        .bind(tiltak_second_move_eval)
        .bind(tinue_length)
        .bind(tinue_avoidance_length)
        .execute(&puzzles_pool)
        .await
        {
            if err.as_database_error().is_some_and(|db_err| db_err.is_unique_violation()) {
                println!("Failed to insert \"{}\" from game ${} into DB due to uniqueness constraint: {}", tps, game_id, err);
                continue;
            }
            println!("Failed to insert \"{}\" from game ${} into DB. Retrying in 1s: {}", tps, game_id, err);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        }
    });

    loop {
        let (game_id, has_been_aborted) = game_receiver.recv().unwrap();

        let analyzed = if has_been_aborted { 2 } else { 1 };

        while let Err(err) = sqlx::query("UPDATE games SET has_been_analyzed = ?1 WHERE id = ?2")
            .bind(analyzed)
            .bind(game_id as u32)
            .execute(&mut db_conn)
            .await
        {
            println!(
                "Failed to update game #{} into DB. Retrying in 1s: {}",
                game_id, err
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

async fn read_non_bot_games(conn: &mut SqliteConnection) -> Option<Vec<PlaytakGame>> {
    let rows: Vec<GameRow> = sqlx::query_as("SELECT * FROM games WHERE has_been_analyzed = 0 AND NOT instr(player_white, \"Bot\") AND NOT instr(player_black, \"Bot\") AND NOT instr(player_white, \"bot\") AND NOT instr(player_black, \"bot\")")
        .fetch_all(conn)
        .await.ok()?;
    rows.into_iter()
        .map(|row| PlaytakGame::try_from(row).ok())
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
        let tps = position.to_fen();

        let tiltak_analysis_shallow = {
            let start_time = Instant::now();
            let result = tiltak_search(position.clone(), 50_000);

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

        let topaz_result = {
            let start_time = Instant::now();
            let result = topaz_search::<S>(&tps);
            stats.topaz_tinue.record(start_time.elapsed());
            result
        };

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
        };

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

            if !last_move_was_tinue && matches!(tinue_avoidance, TopazAvoidanceResult::NoDefense) {
                println!(
                    "Found tinue not identified by Topaz, {} in game #{}",
                    possible_puzzle.previous_tps, game.id
                );
            }

            possible_puzzle.topaz_tinue_avoidance = Some(tinue_avoidance);
        };

        last_move_was_tinue = false;

        match topaz_result {
            TopazResult::NoTinue | TopazResult::AbortedFirst | TopazResult::AbortedSecond(_) => {
                if matches!(
                    possible_puzzle.topaz_tinue_avoidance,
                    Some(TopazAvoidanceResult::Defence(_))
                ) || tiltak_analysis_shallow.score_first > 0.4
                    && tiltak_analysis_shallow.score_second < 0.6
                    && tiltak_analysis_shallow.score_first - tiltak_analysis_shallow.score_second
                        > 0.3
                {
                    let tiltak_start_time = Instant::now();
                    let tiltak_analysis_deep = tiltak_search(position.clone(), 1_000_000);

                    possible_puzzle.tiltak_analysis_deep = Some(tiltak_analysis_deep);

                    stats
                        .tiltak_non_tinue_long
                        .record(tiltak_start_time.elapsed());
                }
            }
            TopazResult::RoadWin => (),
            TopazResult::NonUniqueTinue(_) => last_move_was_tinue = true,
            TopazResult::Tinue(_) => {
                let tiltak_start_time = Instant::now();
                let tiltak_analysis_deep = tiltak_search(position.clone(), 1_000_000);

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

#[derive(sqlx::FromRow, Debug, Clone)]
struct Puzzle {
    game_id: u64,
    tps: String,
    solution: String,
    tiltak_eval: f32,
    tiltak_second_move_eval: f32,
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
            tinue_length: None,
            tinue_avoidance_length: None,
        };
        if let TopazResult::Tinue(moves) = &self.topaz_tinue {
            puzzle.tinue_length = Some(moves.len() as u32);
            puzzle.solution = moves[0].to_string::<S>();
        }
        if let Some(TopazAvoidanceResult::Defence(tinue_avoidance)) = &self.topaz_tinue_avoidance {
            puzzle.tinue_avoidance_length = Some(tinue_avoidance.longest_refutation_length);
            puzzle.solution = tinue_avoidance.defense.to_string::<S>();
        }
        Some(puzzle)
    }

    fn print_stuff<const S: usize>(&self) {
        let Self {
            playtak_game,
            tps,
            topaz_tinue_avoidance,
            tiltak_analysis_shallow,
            ..
        } = &self;

        match &self.topaz_tinue {
            TopazResult::AbortedFirst => {
                println!(
                    "{} was aborted at the first move, game id {}",
                    tps, playtak_game.id
                );
                return;
            }
            TopazResult::AbortedSecond(_) => {
                println!(
                    "{} was aborted at the second move, game id {}",
                    tps, playtak_game.id
                );
                return;
            }
            TopazResult::NonUniqueTinue(_) => {
                println!("Found non-unique tinue");
                return;
            }
            _ => (),
        };

        if let TopazResult::Tinue(topaz_tinue) = self.topaz_tinue.clone() {
            let TiltakResult {
                nodes,
                score_first,
                pv_first,
                score_second,
                pv_second,
            } = self.tiltak_analysis_deep.clone().unwrap();

            if score_second < 0.90 {
                println!(
                    "Tinue in game id {}, {:?} vs {:?}",
                    playtak_game.id, playtak_game.player_white, playtak_game.player_black
                );
                println!("tps: {}", tps);
                println!(
                    "Topaz pv: {}",
                    topaz_tinue
                        .iter()
                        .map(|mv| mv.to_string::<S>())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                println!(
                    "Tiltak first move:  {:.1}%, pv {}",
                    score_first * 100.0,
                    pv_first
                        .iter()
                        .map(|mv| mv.to_string::<S>())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                println!(
                    "Tiltak second move: {:.1}%, pv {}",
                    score_second * 100.0,
                    pv_second
                        .iter()
                        .map(|mv| mv.to_string::<S>())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                println!();
            } else {
                println!("Found boring {}-move tinue", topaz_tinue.len());
            }
        } else if let Some(tiltak_analysis_deep) = self.tiltak_analysis_deep.as_ref() {
            let is_tinue_avoidance = matches!(
                topaz_tinue_avoidance,
                Some(TopazAvoidanceResult::Defence(_))
            );
            if is_tinue_avoidance {
                println!(
                    "Tinue avoidance in game id {}, {:?} vs {:?}",
                    playtak_game.id, playtak_game.player_white, playtak_game.player_black
                );
            } else if tiltak_analysis_deep.score_first > 0.5
                && tiltak_analysis_deep.score_second < 0.5
                && tiltak_analysis_deep.score_first > tiltak_analysis_deep.score_second + 0.3
            {
                println!(
                    "Unique good move in game id {}, {:?} vs {:?}",
                    playtak_game.id, playtak_game.player_white, playtak_game.player_black
                );
            } else {
                println!(
                    "Unique good move rejected after more thinking, shallow scores {:.1}% vs {:.1}% deep scores {:.1}% vs {:.1}%",
                    tiltak_analysis_shallow.score_first * 100.0,
                    tiltak_analysis_shallow.score_second * 100.0,
                    tiltak_analysis_deep.score_first * 100.0,
                    tiltak_analysis_deep.score_second * 100.0
                );
                return;
            }
            println!("tps: {}", tps);
            println!(
                "Tiltak first move:  {:.1}%, pv {}",
                tiltak_analysis_deep.score_first * 100.0,
                tiltak_analysis_deep
                    .pv_first
                    .iter()
                    .map(|mv| mv.to_string::<S>())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!(
                "Tiltak second move: {:.1}%, pv {}",
                tiltak_analysis_deep.score_second * 100.0,
                tiltak_analysis_deep
                    .pv_second
                    .iter()
                    .map(|mv| mv.to_string::<S>())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!();
        }
    }
}

#[derive(Clone)]
struct TinueAvoidance {
    defense: Move,
    refutations: Vec<[Move; 2]>,
    longest_refutation_length: u32,
}

#[derive(Default)]
struct Stats {
    topaz_tinue: TimeTracker,
    topaz_tinue_avoidance: TimeTracker,
    tiltak_tinue: TimeTracker,
    tiltak_non_tinue_short: TimeTracker,
    tiltak_non_tinue_long: TimeTracker,
}

#[derive(Default)]
struct TimeTracker {
    times: Mutex<BTreeSet<Duration>>,
    total_time: Mutex<Duration>,
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
    nodes: u32,
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
            nodes,
            score_first: tree2.best_move().1,
            pv_first: tree2.pv().collect(),
            score_second: score,
            pv_second: tree1.pv().collect(),
        }
    } else {
        TiltakResult {
            nodes,
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

fn topaz_search<const S: usize>(tps: &str) -> TopazResult {
    match S {
        5 => topaz_search_5s(tps),
        6 => topaz_search_6s(tps),
        _ => unimplemented!(),
    }
}

fn topaz_search_5s(tps: &str) -> TopazResult {
    let board = topaz_tak::board::Board5::try_from_tps(tps).unwrap();
    let mut first_tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
        .quiet()
        .limit(5_000_000);
    let is_tinue = first_tinue_search.is_tinue();
    if first_tinue_search.aborted() {
        return TopazResult::AbortedFirst;
    }
    if is_tinue == Some(true) {
        let pv = first_tinue_search.principal_variation();
        let tiltak_pv = pv
            .iter()
            .map(|mv| Move::from_string::<5>(mv.to_ptn::<Board5>().trim_end_matches('*')).unwrap())
            .collect();
        if !pv.is_empty() {
            let mut second_tinue_search = topaz_tak::search::proof::TinueSearch::new(board)
                .quiet()
                .limit(10_000_000)
                .exclude(pv[0]);
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

fn topaz_search_6s(tps: &str) -> TopazResult {
    let board = topaz_tak::board::Board6::try_from_tps(tps).unwrap();
    let mut first_tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
        .quiet()
        .limit(5_000_000);
    let is_tinue = first_tinue_search.is_tinue();
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
            let mut second_tinue_search = topaz_tak::search::proof::TinueSearch::new(board)
                .quiet()
                .limit(10_000_000)
                .exclude(pv[0]);
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
    MultipleDefences,
    NoDefense,
    Defence(TinueAvoidance),
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

    let mut defense = None;
    let mut refutations = vec![];
    let mut longest_refutation_length = 0;

    for mv in legal_moves.iter() {
        let reverse_move = position.do_move(mv.clone());
        let board: Board5 = topaz_tak::board::Board5::try_from_tps(&position.to_fen()).unwrap();
        position.reverse_move(reverse_move);
        let mut tinue_search = topaz_tak::search::proof::TinueSearch::new(board.clone())
            .quiet()
            .limit(1_000_000);
        if tinue_search.aborted() {
            return TopazAvoidanceResult::Aborted;
        }
        if tinue_search.is_tinue() != Some(true) {
            if defense.is_some() {
                return TopazAvoidanceResult::MultipleDefences;
            }
            defense = Some(mv.clone());
        } else if let Some(response) = tinue_search
            .principal_variation()
            .get(0)
            .and_then(|mv| Move::from_string::<S>(&mv.to_ptn::<Board5>()).ok())
        {
            refutations.push([mv.clone(), response]);
            longest_refutation_length =
                longest_refutation_length.max(tinue_search.principal_variation().len() as u32)
        }
    }

    if let Some(mv) = defense {
        TopazAvoidanceResult::Defence(TinueAvoidance {
            defense: mv,
            refutations,
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
    player_white: Option<String>,
    player_black: Option<String>,
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
    rating_change_white: Option<i64>,
    rating_change_black: Option<i64>,
}

impl PlaytakGame {
    pub fn has_standard_piececount(&self) -> bool {
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
        if let (Some(white), Some(black)) = (self.player_white.as_ref(), self.player_black.as_ref())
        {
            BOTS.contains(&white.as_str()) || BOTS.contains(&black.as_str())
        } else {
            false
        }
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
            date_time: DateTime::from_naive_utc_and_offset(
                NaiveDateTime::from_timestamp_opt(row.date / 1000, 0).unwrap(),
                Utc,
            ),
            size: row.size.try_into().map_err(|_| ())?,
            player_white: if row.player_white != "Anon" {
                Some(row.player_white)
            } else {
                None
            },
            player_black: if row.player_black != "Anon" {
                Some(row.player_black)
            } else {
                None
            },
            moves,
            result_string: row.result,
            game_time: Duration::from_secs(row.timertime.try_into().map_err(|_| ())?),
            increment: Duration::from_secs(row.timerinc.try_into().map_err(|_| ())?),
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
            is_rated: match row.unrated {
                0 => true,
                1 => false,
                _ => return Err(()),
            },
            is_tournament: match row.tournament {
                0 => false,
                1 => true,
                _ => return Err(()),
            },
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
            rating_change_white: if row.rating_change_white <= -1000 {
                None
            } else {
                Some(row.rating_change_white)
            },
            rating_change_black: if row.rating_change_black <= -1000 {
                None
            } else {
                Some(row.rating_change_black)
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

#[derive(sqlx::FromRow, Debug, Clone)]
struct GameRow {
    id: i64,
    date: i64,
    size: i64,
    player_white: String,
    player_black: String,
    notation: String,
    result: String,
    timertime: i64,
    timerinc: i64,
    rating_white: i64,
    rating_black: i64,
    unrated: i64,
    tournament: i64,
    komi: i64,
    pieces: i64,
    capstones: i64,
    rating_change_white: i64,
    rating_change_black: i64,
}
