//! Conway's Game of Life on S³ — live terminal animation with pattern selector.
//!
//! Geometric step: alive = carrier(2) on Hopf equator.
//! Rule: half-space test W < 0 ∧ X > 0 on the neighbor product quaternion.
//!
//! Run:  cargo run --example gol_live --release
//! Then press a number key to pick a pattern. Press Q to quit.

use closure_ea::{compose, IDENTITY};
use std::f64::consts::FRAC_1_SQRT_2;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

const ALIVE: [f64; 4] = [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0, 0.0];

// ANSI colors
const GREEN:  &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN:   &str = "\x1b[36m";
const DIM:    &str = "\x1b[2m";
const RESET:  &str = "\x1b[0m";
const BOLD:   &str = "\x1b[1m";

type Grid = Vec<Vec<bool>>;

fn make_grid(rows: usize, cols: usize) -> Grid {
    vec![vec![false; cols]; rows]
}

fn set(grid: &mut Grid, cells: &[(usize, usize)]) {
    for &(r, c) in cells {
        if r < grid.len() && c < grid[0].len() {
            grid[r][c] = true;
        }
    }
}

fn neighbor_product(grid: &Grid, row: usize, col: usize) -> [f64; 4] {
    let rows = grid.len() as i32;
    let cols = grid[0].len() as i32;
    let mut product = IDENTITY;
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if dr == 0 && dc == 0 { continue; }
            let nr = ((row as i32 + dr).rem_euclid(rows)) as usize;
            let nc = ((col as i32 + dc).rem_euclid(cols)) as usize;
            if grid[nr][nc] {
                product = compose(&product, &ALIVE);
            }
        }
    }
    product
}

fn step(grid: &Grid) -> Grid {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut next = make_grid(rows, cols);
    const EPS: f64 = 1e-9;
    for r in 0..rows {
        for c in 0..cols {
            let p = neighbor_product(grid, r, c);
            next[r][c] = if grid[r][c] {
                p[0] < EPS && p[1] > EPS       // survive: W ≤ 0, X > 0
            } else {
                p[0] < -EPS && p[1] > EPS      // born:    W < 0, X > 0
            };
        }
    }
    next
}

fn count_alive(grid: &Grid) -> usize {
    grid.iter().flat_map(|r| r.iter()).filter(|&&c| c).count()
}

fn render(grid: &Grid, gen: usize, alive: usize, pattern_name: &str) -> String {
    let cols = grid[0].len();
    let mut out = String::new();

    // header
    out.push_str(&format!(
        "{BOLD}  S³ Game of Life{RESET}  {DIM}│{RESET}  \
        {CYAN}pattern: {}{RESET}  {DIM}│{RESET}  \
        gen: {BOLD}{:>5}{RESET}  {DIM}│{RESET}  \
        alive: {GREEN}{:>4}{RESET}\n",
        pattern_name, gen, alive
    ));
    out.push_str(&format!(
        "  {DIM}alive = carrier(2) ∈ Hopf equator  │  \
        rule: W<0 ∧ X>0 on ∏ neighbors{RESET}\n"
    ));

    // top border
    out.push_str(&format!("  {DIM}┌{}┐{RESET}\n", "─".repeat(cols)));

    for row in grid.iter() {
        out.push_str(&format!("  {DIM}│{RESET}"));
        for &cell in row.iter() {
            if cell {
                out.push_str(&format!("{YELLOW}█{RESET}"));
            } else {
                out.push(' ');
            }
        }
        out.push_str(&format!("{DIM}│{RESET}\n"));
    }

    // bottom border
    out.push_str(&format!("  {DIM}└{}┘{RESET}\n", "─".repeat(cols)));
    out.push_str(&format!("  {DIM}Ctrl+C to stop{RESET}\n"));
    out
}

// ── Patterns ─────────────────────────────────────────────────────────────────

fn pattern_rpentomino(grid: &mut Grid, r: usize, c: usize) {
    // R-pentomino: 5 cells, chaotic for 1103 generations
    set(grid, &[
        (r,   c+1),(r,   c+2),
        (r+1, c),  (r+1, c+1),
        (r+2, c+1),
    ]);
}

fn pattern_acorn(grid: &mut Grid, r: usize, c: usize) {
    // Acorn: 7 cells, chaotic for 5206 generations before stabilising
    set(grid, &[
        (r,   c+1),
        (r+1, c+3),
        (r+2, c),  (r+2, c+1), (r+2, c+4), (r+2, c+5), (r+2, c+6),
    ]);
}

fn pattern_glider(grid: &mut Grid, r: usize, c: usize) {
    set(grid, &[
        (r,   c+1),
        (r+1, c+2),
        (r+2, c),  (r+2, c+1), (r+2, c+2),
    ]);
}


fn pattern_pulsar(grid: &mut Grid, r: usize, c: usize) {
    // Pulsar: period-3 oscillator, 48 cells, 13×13 bounding box
    let cells: &[(usize, usize)] = &[
        (0,2),(0,3),(0,4),   (0,8),(0,9),(0,10),
        (2,0),(2,5),(2,7),(2,12),
        (3,0),(3,5),(3,7),(3,12),
        (4,0),(4,5),(4,7),(4,12),
        (5,2),(5,3),(5,4),   (5,8),(5,9),(5,10),
        (7,2),(7,3),(7,4),   (7,8),(7,9),(7,10),
        (8,0),(8,5),(8,7),(8,12),
        (9,0),(9,5),(9,7),(9,12),
        (10,0),(10,5),(10,7),(10,12),
        (12,2),(12,3),(12,4),(12,8),(12,9),(12,10),
    ];
    set(grid, &cells.iter().map(|&(dr,dc)| (r+dr, c+dc)).collect::<Vec<_>>());
}

fn pattern_diehard(grid: &mut Grid, r: usize, c: usize) {
    // Diehard: dies completely after exactly 130 generations
    set(grid, &[
        (r,   c+6),
        (r+1, c),  (r+1, c+1),
        (r+2, c+1),(r+2, c+5),(r+2, c+6),(r+2, c+7),
    ]);
}

fn pattern_gun(grid: &mut Grid, r: usize, c: usize) {
    // Gosper Glider Gun — canonical coordinates from LifeWiki RLE.
    // Period 30: fires one glider every 30 generations indefinitely.
    let cells: &[(usize, usize)] = &[
        (0,24),
        (1,22),(1,24),
        (2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
        (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),
        (4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
        (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,22),(5,24),
        (6,10),(6,16),(6,24),
        (7,11),(7,15),
        (8,12),(8,13),
    ];
    set(grid, &cells.iter().map(|&(dr,dc)| (r+dr, c+dc)).collect::<Vec<_>>());
}

// ── LWSS helper ───────────────────────────────────────────────────────────────

fn pattern_lwss(grid: &mut Grid, r: usize, c: usize) {
    // Lightweight spaceship — moves right at c/2 speed. Phase 0.
    set(grid, &[
        (r,   c+1),(r,   c+4),
        (r+1, c),
        (r+2, c),              (r+2, c+4),
        (r+3, c),(r+3, c+1),(r+3, c+2),(r+3, c+3),
    ]);
}

fn pattern_fleet(grid: &mut Grid) {
    // 5 lightweight spaceships in V-formation heading right.
    // All move at identical speed — the chevron shape holds forever.
    let rows = grid.len();
    let cr = rows / 2;
    pattern_lwss(grid, cr.saturating_sub(2), 30);   // tip (rightmost)
    pattern_lwss(grid, cr.saturating_sub(9), 18);   // inner wing top
    pattern_lwss(grid, cr + 5,               18);   // inner wing bottom
    pattern_lwss(grid, cr.saturating_sub(16), 6);   // outer wing top
    pattern_lwss(grid, cr + 12,               6);   // outer wing bottom
}

fn pattern_d8_mandala(grid: &mut Grid) {
    // 8 oriented acorns arranged with full D8 symmetry.
    //
    // The old mandala used a tidy oscillator, which stayed alive but looked
    // too polite. This version uses the acorn methuselah (lifespan 5206) as
    // the seed motif and places all 8 dihedral transforms of one off-center
    // copy around the board. Because the motif itself is asymmetric, the full
    // 8-copy configuration keeps exact D8 symmetry while erupting into a much
    // richer, longer-lived geometric snowflake.
    //
    // Acorn relative cells, centered roughly at its active core:
    //   .O.....
    //   ...O...
    //   OO..OOO
    let acorn: &[(i32, i32)] = &[
        (-1, -2),
        ( 0,  0),
        ( 1, -3), ( 1, -2), ( 1,  1), ( 1,  2), ( 1,  3),
    ];
    let cr = (grid.len() / 2) as i32;
    let cc = (grid[0].len() / 2) as i32;
    let (or, oc): (i32, i32) = (30, 12);

    let transforms: &[(i32, i32)] = &[
        ( or,  oc), (-oc,  or), (-or, -oc), ( oc, -or),
        (-or,  oc), ( or, -oc), ( oc,  or), (-oc, -or),
    ];

    let mut cells = Vec::new();
    for &(base_r, base_c) in transforms {
        for &(pr, pc) in acorn {
            let tr = if base_r ==  or && base_c ==  oc { pr }       // identity
            else if base_r == -oc && base_c ==  or { -pc }          // rot 90
            else if base_r == -or && base_c == -oc { -pr }          // rot 180
            else if base_r ==  oc && base_c == -or { pc }           // rot 270
            else if base_r == -or && base_c ==  oc { -pr }          // reflect x
            else if base_r ==  or && base_c == -oc { pr }           // reflect y
            else if base_r ==  oc && base_c ==  or { pc }           // reflect diag y=x
            else { -pc };                                           // reflect diag y=-x

            let tc = if base_r ==  or && base_c ==  oc { pc }
            else if base_r == -oc && base_c ==  or { pr }
            else if base_r == -or && base_c == -oc { -pc }
            else if base_r ==  oc && base_c == -or { -pr }
            else if base_r == -or && base_c ==  oc { pc }
            else if base_r ==  or && base_c == -oc { -pc }
            else if base_r ==  oc && base_c ==  or { pr }
            else { -pr };

            let gr = cr + base_r + tr;
            let gc = cc + base_c + tc;
            if gr >= 0 && gc >= 0 && (gr as usize) < grid.len() && (gc as usize) < grid[0].len() {
                cells.push((gr as usize, gc as usize));
            }
        }
    }
    set(grid, &cells);
}

fn pattern_battle(grid: &mut Grid) {
    // Two Gosper Glider Guns aimed at each other so gliders hit each them.
    //
    let cells: &[(usize, usize)] = &[
        (0,24),(1,22),(1,24),
        (2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
        (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),
        (4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
        (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,22),(5,24),
        (6,10),(6,16),(6,24),
        (7,11),(7,15),(8,12),(8,13),
    ];
    let (max_dr, max_dc) = (8usize, 35usize);

    // Gun A — upper area, standard orientation.
    // Glider exit ≈ (a_r+5, a_c+22) = (8, 77).  r+c = 85.  Fires SW.
    let a_r = 3usize;
    let a_c = 55usize;
    set(grid, &cells.iter().map(|&(dr,dc)| (dr + a_r, dc + a_c)).collect::<Vec<_>>());

    // Gun B — lower area, 180° rotation.
    // Glider exit ≈ (b_r+3, b_c+13) = (41, 44).  r+c = 85.  Fires NE.
    // Same anti-diagonal as Gun A → streams collide at ≈ (25, 60).
    let b_r = 38usize;
    let b_c = 31usize;
    set(grid, &cells.iter().map(|&(dr,dc)| {
        (max_dr - dr + b_r, max_dc - dc + b_c)
    }).collect::<Vec<_>>());
}

fn pattern_random(grid: &mut Grid, density: f64) {
    // Simple LCG for deterministic "random" without importing rand
    let mut seed: u64 = 0xDEAD_BEEF_1337_4242;
    for row in grid.iter_mut() {
        for cell in row.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *cell = (seed >> 33) as f64 / (u32::MAX as f64) < density;
        }
    }
}

fn build_grid(name: &str) -> (Grid, String) {
    match name {
        "acorn" => {
            let mut g = make_grid(32, 70);
            pattern_acorn(&mut g, 14, 31);
            (g, "acorn — 7 cells, 5206 generations of chaos".to_string())
        }
        "gun" => {
            // 50-row grid: gliders take ~200 gen to wrap vertically; safe for 180 gen
            let mut g = make_grid(50, 110);
            pattern_gun(&mut g, 20, 35);
            (g, "Gosper glider gun — period 30, fires a glider every 30 gen".to_string())
        }
        "pulsar" => {
            let mut g = make_grid(26, 60);
            // Three pulsars side by side
            pattern_pulsar(&mut g, 6, 5);
            pattern_pulsar(&mut g, 6, 24);
            pattern_pulsar(&mut g, 6, 43);
            (g, "pulsar × 3 — period-3 oscillator, 48 cells each".to_string())
        }
        "fleet" => {
            let mut g = make_grid(40, 90);
            pattern_fleet(&mut g);
            (g, "LWSS chevron — 5 lightweight spaceships in V-formation, moves forever".to_string())
        }
        "mandala" => {
            // Square grid — D4 symmetry requires rows == cols on toroidal grid
            let mut g = make_grid(128, 128);
            pattern_d8_mandala(&mut g);
            (g, "D8 mandala — 8 acorns on a dihedral ring, long-lived geometric snowflake chaos".to_string())
        }
        "battle" => {
            let mut g = make_grid(50, 120);
            pattern_battle(&mut g);
            (g, "battle — two glider guns battling each other".to_string())
        }
        "diehard" => {
            let mut g = make_grid(20, 50);
            pattern_diehard(&mut g, 8, 20);
            (g, "diehard — vanishes completely after exactly 130 generations".to_string())
        }
        "random" => {
            let mut g = make_grid(30, 70);
            pattern_random(&mut g, 0.30);
            (g, "random soup — 30% density, watch it self-organise".to_string())
        }
        _ => {
            let mut g = make_grid(30, 66);
            pattern_rpentomino(&mut g, 13, 31);
            pattern_glider(&mut g, 2, 2);
            pattern_glider(&mut g, 2, 55);
            let (last, wide) = (g.len(), g[0].len());
            set(&mut g, &[(last-5, 10),(last-5, 11),(last-5, 12)]);
            set(&mut g, &[(3, wide-7),(4, wide-7),(3, wide-6),(4, wide-6)]);
            (g, "r-pentomino + gliders — 1103 generations of chaos".to_string())
        }
    }
}

// ── Raw terminal input (no external crates) ──────────────────────────────────

/// Switch terminal to cbreak mode: keypresses arrive immediately, no echo.
fn raw_on() {
    let _ = std::process::Command::new("stty")
        .args(["-F", "/dev/tty", "-echo", "cbreak"])
        .status();
}

/// Restore normal terminal mode.
fn raw_off() {
    let _ = std::process::Command::new("stty")
        .args(["-F", "/dev/tty", "echo", "-cbreak"])
        .status();
}

/// Read a single byte from /dev/tty (blocking).
fn read_key() -> u8 {
    use std::io::Read;
    let mut tty = std::fs::File::open("/dev/tty").expect("cannot open /dev/tty");
    let mut buf = [0u8; 1];
    tty.read_exact(&mut buf).ok();
    buf[0]
}

// ── Menu ─────────────────────────────────────────────────────────────────────

const MENU: &[(&str, &str, usize)] = &[
    ("1", "R-pentomino",  600),
    ("2", "Acorn",        800),
    ("3", "Glider gun",   180),
    ("4", "Pulsar",       300),
    ("5", "Fleet",        600),
    ("6", "Diehard",      145),
    ("7", "Random soup",  400),
    ("8", "Mandala",      800),
    ("9", "Battle",       500),
    ("0", "Showcase",       0),  // 0 = cycle indefinitely
];

const MENU_KEYS: &[&str] = &[
    "rpentomino", "acorn", "gun", "pulsar", "fleet", "diehard", "random", "mandala", "battle", "showcase",
];

fn show_menu() {
    print!("\x1b[2J\x1b[H");
    println!();
    println!("  {BOLD}{CYAN}Conway's Game of Life  ×  S³ Geometric Computer{RESET}");
    println!();
    println!("  {DIM}alive = carrier(2) on Hopf equator  │  rule: W<0 ∧ X>0 on ∏ neighbors{RESET}");
    println!();
    println!("  {DIM}────────────────────────────────────────────────────{RESET}");
    println!();
    for (i, (key, name, _)) in MENU.iter().enumerate() {
        let label = if i == 9 {
            format!("  {BOLD}[{key}]{RESET}  {name:<16}  {DIM}cycles through all patterns{RESET}")
        } else {
            let desc = match i {
                0 => "chaos for 1103 generations",
                1 => "7 cells, chaos for 5206 generations",
                2 => "fires a glider every 30 gen, indefinitely",
                3 => "period-3 oscillator, three pulsars in sync",
                4 => "5 LWSSs in V-formation — holds shape forever",
                5 => "vanishes completely after exactly 130 gen",
                6 => "30% random density — self-organises",
                7 => "8 acorns in full D8 symmetry — blooms for a long time",
                8 => "two glider guns face each other — infinite collision growth",
                _ => "",
            };
            format!("  {BOLD}[{key}]{RESET}  {name:<16}  {DIM}{desc}{RESET}")
        };
        println!("{label}");
    }
    println!();
    println!("  {DIM}────────────────────────────────────────────────────{RESET}");
    println!();
    println!("  {BOLD}Press 1–9 or 0 to run a pattern.  Q to quit.{RESET}");
    println!();
    io::stdout().flush().unwrap();
}

fn run_pattern(key: &str, max_gen: usize) {
    let delay = Duration::from_millis(75);

    if key == "showcase" {
        let sequence: &[(&str, usize)] = &[
            ("mandala",    300),
            ("battle",     400),
            ("gun",        150),
            ("fleet",      300),
            ("rpentomino", 300),
            ("pulsar",     200),
            ("acorn",      250),
            ("diehard",    145),
            ("random",     200),
        ];
        for &(pat, n) in sequence.iter().cycle() {
            let (mut grid, desc) = build_grid(pat);
            for gen in 0..n {
                let alive = count_alive(&grid);
                print!("\x1b[H{}", render(&grid, gen, alive, &desc));
                io::stdout().flush().unwrap();
                grid = step(&grid);
                thread::sleep(delay);
            }
        }
    } else {
        let (mut grid, desc) = build_grid(key);
        let mut gen = 0usize;
        loop {
            let alive = count_alive(&grid);
            print!("\x1b[H{}", render(&grid, gen, alive, &desc));
            io::stdout().flush().unwrap();
            grid = step(&grid);
            gen += 1;
            if max_gen > 0 && gen >= max_gen { break; }
            thread::sleep(delay);
        }
        // Linger on last frame a moment before returning to menu
        thread::sleep(Duration::from_millis(1200));
    }
}

fn main() {
    // Hide cursor
    print!("\x1b[?25l");
    io::stdout().flush().unwrap();

    // Restore cursor + terminal on exit (best-effort)
    let _ = std::panic::catch_unwind(|| {});

    loop {
        raw_off();
        show_menu();
        raw_on();
        let key = read_key();
        raw_off();

        match key {
            b'q' | b'Q' => {
                print!("\x1b[?25h\x1b[2J\x1b[H");
                io::stdout().flush().unwrap();
                break;
            }
            b'1'..=b'9' => {
                let idx = (key - b'1') as usize;
                let (_, max_gen) = (MENU_KEYS[idx], MENU[idx].2);
                print!("\x1b[2J");
                io::stdout().flush().unwrap();
                run_pattern(MENU_KEYS[idx], max_gen);
            }
            b'0' => {
                print!("\x1b[2J");
                io::stdout().flush().unwrap();
                run_pattern("showcase", 0);
            }
            _ => {}  // ignore anything else, redraw menu
        }
    }

    // Restore terminal on normal exit
    print!("\x1b[?25h");
    io::stdout().flush().unwrap();
}
