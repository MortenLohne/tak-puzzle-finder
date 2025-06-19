use board_game_traits::Position as PositionTrait;
use pgn_traits::PgnPosition;
use tiltak::position::{Move, Position};

use crate::{
    PuzzleF, PuzzleRoot, Stats, find_followup, find_followups, find_full_puzzles,
    followups::{extract_possible_full_tinues, find_desperado_defense_lines},
};

#[test]
pub fn find_followups_test() {
    let puzzle_root: PuzzleRoot<6> = PuzzleRoot {
        playtak_game_id: 594670,
        tps: "212,x,1,1,1,1/x3,1,x2/12,2S,11C,x,2S,x/21S,2,21212C,1S,221,x/x,2,2,2,1,2/2,2,x,2,1221S,x 1 30".to_string(),
        solution: Move::from_string("2c4>11").unwrap(),
        tinue_length: 7
    };

    let stats = &Stats::default();
    let followups = find_followup(&puzzle_root, &stats);
    println!("{} followups found", followups.len());
    for tinue_followup in followups.iter().filter_map(|tinue_followup| {
        if let PuzzleF::UniqueTinue(tinue) = tinue_followup {
            Some(tinue)
        } else {
            None
        }
    }) {
        println!(
            "Followup solution: {} {}, tinue length {}, 0 komi score {}, 2 komi score: {}",
            tinue_followup.parent_move,
            tinue_followup.solution,
            tinue_followup.tinue_length,
            tinue_followup.score_0komi(),
            tinue_followup.score_2komi()
        );
    }
    assert!(false);
}

// Requires manual interaction, and running tests with -- --nocapture
#[test]
fn full_test() {
    let db_path = "src/tests/test.db";
    let conn = rusqlite::Connection::open(db_path).unwrap();
    // Reset the test database to a clean state
    conn.execute("UPDATE puzzles SET followups_analyzed = 0", [])
        .unwrap();
    conn.execute("DROP TABLE IF EXISTS tinue_followups", [])
        .unwrap();
    conn.execute("DROP TABLE IF EXISTS full_tinue_puzzles", [])
        .unwrap();
    conn.execute("DROP TABLE IF EXISTS road_win_followups", [])
        .unwrap();

    find_followups::<6>(db_path);
    find_full_puzzles::<6>(&db_path)
}

#[test]
fn find_full_tinue_test1() {
    find_full_tinue_prop::<6>(
        "212,x,1,1,1,1/x3,1,x2/12,2S,11C,x,2S,x/21S,2,21212C,1S,221,x/x,2,2,2,1,2/2,2,x,2,1221S,x 1 30",
        "2c4>11*",
        &["2c4>11*", "d2>", "3e1+", "d1>", "4e2-", "f2<", "5e1+"],
    );
}

#[test]
fn find_full_tinue_test2() {
    find_full_tinue_prop::<6>(
        "1,1,221S,1,12,x/x,2,2,x,12,1/x2,1,2,212C,x/x,2,2221S,1,1S,2/2,2,2,121,x2/2,x2,1121C,x,1 1 36",
        "3d1+",
        &["3d1+", "Sd5", "e1"],
    );
}

#[test]
fn find_full_tinue_test3() {
    find_full_tinue_prop::<6>(
        "2,x2,1,x2/2,1S,x,1,x2/221S,2,2,1121C,x2/x,12C,1S,112,x,212S/2,2,2,2,1S,1/2,x4,1 1 22",
        "3d4-12",
        &[
            "3d4-12", "Sd1", "d2+", "c4>", "4d3+", "3f3<12", "d4-", "Sf3", "d3+",
        ],
    );
}

fn find_full_tinue_prop<const S: usize>(
    tps: &str,
    solution: &str,
    possible_solution: &[&'static str],
) {
    let mut position: Position<S> = Position::from_fen(tps).unwrap();
    let mv = Move::from_string(solution).unwrap();
    assert!(position.move_is_legal(mv), "Move is not legal in position");
    position.do_move(mv);

    let stats = Stats::default();

    let candidate_tinue = extract_possible_full_tinues(position, &stats);

    let possible_solution: Vec<Move<S>> = possible_solution
        .into_iter()
        .map(|move_string| Move::from_string(move_string).unwrap())
        .collect();

    for (tinue, goes_to_road) in candidate_tinue.solutions.iter() {
        println!(
            "Tinue goes to road: {}, solution: {}",
            goes_to_road,
            tinue
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    assert!(
        candidate_tinue
            .solutions
            .iter()
            .any(|(solution, _)| solution.starts_with(&possible_solution)),
        "Solution tinue not found in full tinues"
    );
}

#[test]
fn find_desperado_defense() {
    let mut position: Position<6> = Position::from_fen(
        "x2,2,1,221,1/2,2,2S,x,1221,x/x,1S,12,2C,2S,12S/x,2,1,21C,221,x/x2,1221,221,1,x/1,1,1,x,2S,x 2 32",
    ).unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(tinue_lines.is_some(), "No desperado defense lines found");
}

#[test]
fn no_desperado_defense() {
    let mut position: Position<6> = Position::from_fen(
        "2,2,x,1,2S,1/x,2,x,1,1,1/2,x,21221,1,2,x/x,2221,2,11,x,1112S/2,12111112C,2,221C,1,x/221S,2,2,2,x,1 2 44",
    ).unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(
        tinue_lines.is_none(),
        "Desperado defense lines found when none expected"
    );
}

#[test]
fn find_desperado_defense2() {
    // This position's defensive moves require two-length spreads with capstone to refute
    let mut position: Position<6> = Position::from_fen(
        "x,1,x,2,2,x/1,1,11121C,1,121S,1/2,1,12,12,1,x/2,1,x,2,2,x/2,221S,12C,1S,2,2/2,1,2,212,2,1 2 29",
    ).unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(tinue_lines.is_some());
}

#[test]
fn find_desperado_defense3() {
    let mut position: Position<6> = Position::from_fen(
        "2,2212C,1,2,2,2/x2,11C,212112,2,x/x,1,211,21,21,x/x3,1,x,2S/x,112S,2,x2,1/1,1,1,21S,x2 1 34",
    ).unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(tinue_lines.is_some());
}

#[test]
fn find_desperado_defense4() {
    // This position's defensive moves (25... 2b4>11) require a non-pure spread
    // from a flat they just gave us to refute
    let mut position: Position<6> = Position::from_fen(
        "11C,x,1,1,x2/1122C,121,1S,1,x2/2,12,x,1,x2/12,112,x,1,1,1/2,2,x4/12,x5 2 25",
    )
    .unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(tinue_lines.is_some());
}

#[test]
fn no_desperado_defense2() {
    let mut position: Position<6> = Position::from_fen(
        "2,2,x4/x,2,x4/2,2,2C,2,21,1/2,21C,12,11,1,2/1,12121,x,1,x2/12,x,1,x2,1 2 20",
    )
    .unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(
        tinue_lines.is_none(),
        "Desperado defense lines found when none expected"
    );
}

#[test]
fn no_desperado_defense3() {
    // 37... b5< is not a desperado defense, and requires a more difficult refutation
    let mut position: Position<6> = Position::from_fen(
        "2,2,112S,1,1S,221/1,2,121C,21,21,21/1,12,x2,2,2/1,1,11112C,1,2,2/2S,21,x,112,2,x/2,21,x,2,x,1 2 37",
    )
    .unwrap();
    let tinue_lines = find_desperado_defense_lines(&mut position);
    assert!(
        tinue_lines.is_none(),
        "Desperado defense lines found when none expected"
    );
}
