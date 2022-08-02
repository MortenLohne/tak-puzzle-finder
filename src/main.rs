use std::time::Duration;

use board_game_traits::{GameResult, Position as PositionTrait};
use chrono::{DateTime, NaiveDateTime, Utc};
use pgn_traits::PgnPosition;
use sqlx::{Connection, Executor, SqliteConnection, Row};
use tiltak::position::{self, Komi, Move, Position};

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    // Create a connection pool
    //  for MySQL, use MySqlPoolOptions::new()
    //  for SQLite, use SqlitePoolOptions::new()
    //  etc.
    let mut conn = SqliteConnection::connect("sqlite:games_anon.db").await?;
    let games = read_non_bot_games(&mut conn).await.unwrap();

    let mut num_games = 0;
    let mut num_bot_games = 0;
    let mut num_illegal = 0;
    let mut num_non_standard_piececounts = 0;
    let mut num_wrong_size = 0;

    for game in games {
        if game.size == 6 {
            println!("Processing {:?} vs {:?}", game.player_white, game.player_black);
            let mut position = <Position<6>>::start_position_with_komi(game.komi);

            for mv in &game.moves {
                process_position6(&position.to_fen());
                position.do_move(mv.clone());
            }
        }
        if game.size > 6 {
            num_wrong_size += 1;
            continue;
        }
        num_games += 1;
        if game.is_bot_game() {
            println!("{:?} vs {:?}", game.player_white, game.player_black);
            num_bot_games += 1;
        }
        if !game.has_standard_piececount() {
            num_non_standard_piececounts += 1;
        }
        else if !game.game_is_legal() {
            num_illegal += 1;
        }
    }

    println!("{} games total, {} bot games, {} wrong size, {} nonstandard, {} with illegal moves", num_games, num_bot_games, num_wrong_size, num_non_standard_piececounts, num_illegal);

    Ok(())
}

async fn read_non_bot_games(conn: &mut SqliteConnection) -> Option<Vec<PlaytakGame>> {
    let rows: Vec<GameRow> = sqlx::query_as("SELECT * FROM games WHERE NOT instr(player_white, \"Bot\") AND NOT instr(player_black, \"Bot\") AND NOT instr(player_white, \"bot\") AND NOT instr(player_black, \"bot\")")
        .fetch_all(conn)
        .await.ok()?;
    rows.into_iter().map(|row| PlaytakGame::try_from(row).ok()).collect()
}

fn process_position6(tps: &str) {
    let board = topaz_tak::board::Board6::try_from_tps(tps).unwrap();
    let mut tinue_search = topaz_tak::search::proof::TinueSearch::new(board);
    println!("Is tinue: {:?}", tinue_search.is_tinue());

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
        if let (Some(white), Some(black)) = (self.player_white.as_ref(), self.player_black.as_ref()) {
            BOTS.contains(&white.as_str()) || BOTS.contains(&black.as_str())
        }
        else {
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
            return true;
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
            player_white: if row.player_white != "Anon" { Some(row.player_white) } else { None },
            player_black: if row.player_black != "Anon" { Some(row.player_black) } else { None },
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
