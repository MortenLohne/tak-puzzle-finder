use tiltak::position::Move;

use crate::{PuzzleF, PuzzleRoot, Stats, find_followup, find_followups, find_full_puzzles};

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
