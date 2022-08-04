use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::{sync::Mutex, time::Duration};

use rayon::prelude::*;

use board_game_traits::Position as PositionTrait;
use chrono::{DateTime, NaiveDateTime, Utc};
use pgn_traits::PgnPosition;
use sqlx::{Connection, SqliteConnection};
use tiltak::position::{self, Komi, Move, Position};
use tiltak::search;
use topaz_tak::board::Board6;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    // Create a connection pool
    //  for MySQL, use MySqlPoolOptions::new()
    //  for SQLite, use SqlitePoolOptions::new()
    //  etc.
    let mut conn = SqliteConnection::connect("sqlite:games_anon.db").await?;
    let games = read_non_bot_games(&mut conn).await.unwrap();

    let times = Mutex::new(BTreeSet::new());
    let time_sum = Mutex::new(Duration::ZERO);
    let games_processed = AtomicU64::new(0);

    games.par_iter()
        .filter(|game| game.size == 6 && !game.is_bot_game() && game.game_is_legal() && game.has_standard_piececount() && game.player_white.is_some() && game.player_black.is_some())
        .for_each(|game| {
            let mut position = <Position<6>>::start_position_with_komi(game.komi);

            for mv in &game.moves {
                let tps = position.to_fen();
                let topaz_start_time = Instant::now();
                let topaz_result = process_position6(&tps);
                {
                    let topaz_time_taken = topaz_start_time.elapsed();
                    times.lock().unwrap().insert(topaz_time_taken);
                    *time_sum.lock().unwrap() += topaz_time_taken;
                }
                match topaz_result {
                    TopazResult::NoTinue => (),
                    TopazResult::RoadWin => (),
                    TopazResult::AbortedFirst => println!("{} was aborted at the first move, game id {}", tps, game.id),
                    TopazResult::AbortedSecond(_) => println!("{} was aborted at the second move, game id {}", tps, game.id),
                    TopazResult::NonUniqueTinue(_) => println!("Found non-unique tinue"),
                    TopazResult::Tinue(moves) => {
                        const NODES: u32 = 1_000_000;
                        let settings = search::MctsSetting::default().arena_size_for_nodes(NODES);
                        let mut tree = search::MonteCarloTree::with_settings(position.clone(), settings);
                        for _ in 0..NODES {
                            tree.select();
                        }
                        let (best_move, score) = tree.best_move();

                        let settings2 = search::MctsSetting::default().arena_size_for_nodes(NODES).exclude_moves(vec![best_move.clone()]);
                        let mut tree2 = search::MonteCarloTree::with_settings(position.clone(), settings2);
                        for _ in 0..NODES {
                            tree2.select();
                        }
                        let (best_move2, score2) = tree2.best_move();

                        if score.min(score2) < 0.90 {
                            println!("Game id {}, {:?} vs {:?}", game.id, game.player_white, game.player_black);
                            println!("tps: {}", position.to_fen());
                            print!("Topaz pv: ");
                            for mv in moves.iter() {
                                print!("{} ", mv.to_string::<6>());
                            }
                            println!("\nTiltak: {}, {:.1}%, second move {}, {:.1}%\n", 
                                best_move.to_string::<6>(),
                                score * 100.0,
                                best_move2.to_string::<6>(),
                                score2 * 100.0)
                        }
                        else {
                            println!("Found boring {}-move tinue", moves.len());
                        }
                    }
                }

                position.do_move(mv.clone());
            }
            if games_processed.fetch_add(1, Ordering::SeqCst) % 100 == 0 {
                let times_unlocked = times.lock().unwrap();
                println!("Checked {}/{} games, {} moves, {:.2}s average time, longest {:.2}s, top 50% {:.2}s, top 1.6% {:.2}s, top 0.8% {:.2}s, top 0.4% {:.2}s, top 0.2% {:.2}s, top 0.1% {:.2}s", 
                    games_processed.load(Ordering::SeqCst),
                    games.len(),
                    times_unlocked.len(),
                    time_sum.lock().unwrap().as_secs_f32() / times_unlocked.len() as f32,
                    times_unlocked.iter().last().cloned().unwrap_or_default().as_secs_f32(),
                    times_unlocked.iter().nth(times_unlocked.len() / 2).cloned().unwrap_or_default().as_secs_f32(),
                    times_unlocked.iter().nth(984 * times_unlocked.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
                    times_unlocked.iter().nth(992 * times_unlocked.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
                    times_unlocked.iter().nth(996 * times_unlocked.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
                    times_unlocked.iter().nth(998 * times_unlocked.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
                    times_unlocked.iter().nth(999 * times_unlocked.len() / 1000).cloned().unwrap_or_default().as_secs_f32(),
                    );
            }

    });

    Ok(())
}

async fn read_non_bot_games(conn: &mut SqliteConnection) -> Option<Vec<PlaytakGame>> {
    let rows: Vec<GameRow> = sqlx::query_as("SELECT * FROM games WHERE NOT instr(player_white, \"Bot\") AND NOT instr(player_black, \"Bot\") AND NOT instr(player_white, \"bot\") AND NOT instr(player_black, \"bot\")")
        .fetch_all(conn)
        .await.ok()?;
    rows.into_iter()
        .map(|row| PlaytakGame::try_from(row).ok())
        .collect()
}

enum TopazResult {
    NoTinue,
    RoadWin,
    Tinue(Vec<Move>),
    AbortedFirst,
    AbortedSecond(Vec<Move>),
    NonUniqueTinue(Vec<Move>),
}

fn process_position6(tps: &str) -> TopazResult {
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

#[derive(Debug)]
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
        const BOTS: &'static [&'static str] = &[
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
                if position.game_result() != None {
                    return false;
                }
                position.generate_moves(&mut legal_moves);
                if !legal_moves.contains(&mv) {
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
            date_time: DateTime::from_utc(NaiveDateTime::from_timestamp(row.date / 1000, 0), Utc),
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
            .map(|mv| Move::from_string_playtak::<S>(mv))
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
