//! The Gray Game on S³ — three interference regimes.
//!
//! MODE 1  "Spectrum"  (key 1)
//!   Pure coherence / integration.  15 random patches → whole spectrum emerges.
//!   Vector-mean rule: every cell drifts toward its neighborhood consensus.
//!   Entropy decreases.  Ends in a smooth gradient — one dominant frequency.
//!
//! MODE 2  "Resonance"  (key 2)
//!   6 primary sectors (60° slices).  Two-zone rule:
//!     stable core  (coh > 0.72, n ≤ 6) → mean rule, domain holds
//!     boundary     (coh ∈ [0.28, 0.72], n ∈ [2,6]) → Hamilton product of the
//!                  MOST DIFFERENT alive-neighbor pair  →  new frequency.
//!     overcrowded  (n > 6) or incoherent (coh < 0.28) → die
//!   Only one pair is composed at each boundary cell, so the product is always a
//!   single clean resonance event — not composed-chaos.  6 primaries produce 15
//!   distinct secondaries; those produce tertiaries; the wheel fills in.
//!
//! MODE 3  "Edge"  (key 3)
//!   The critical point.  Same 6-sector seed at lower density.
//!   Rule: blend the mean carrier and the Hamilton product of all alive neighbors
//!   at ratio β = 0.38.
//!     β = 0  →  pure mean (Mode 1)
//!     β = 1  →  pure product (Mode 2 chaos)
//!     β = 0.38 → neither side wins: domains form but never freeze,
//!                boundaries churn but never dissolve.
//!   Long-lived, persistent, evolving structure.
//!
//! COLOR
//!   Rotation axis (X,Y,Z)/|(X,Y,Z)| ∈ S² → RGB:
//!     R = X  salience   — antisymmetric commutator; what demands attention
//!     G = Y  total      — the whole field, prior, everything; G→1 at saturation
//!     B = Z  unknown    — what has not been integrated into the model
//!
//!   Known = G − B  (total minus unknown = what has been learned).
//!   Yellow (high G, low B) = high known, low unknown = learned, predicted.
//!   Blue   (low G, high B) = low total, high unknown  = novel, unlearned.
//!   Brightness = √(σ/(π/2)).
//!
//! Controls:  1 / 2 / 3 = mode    SPACE = reseed    Q = quit

use closure_ea::{compose, sigma, IDENTITY};
use std::f64::consts::{PI, FRAC_PI_2};
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

const ROWS: usize = 38;
const COLS: usize = 100;

// ── Mode 1 ────────────────────────────────────────────────────────────────────
const M1_BIRTH_COH: f64 = 0.52;
const M1_SURV_COH:  f64 = 0.28;
const M1_MIN:       usize = 2;

// ── Mode 2 ────────────────────────────────────────────────────────────────────
const M2_STABLE_COH: f64 = 0.72;  // above → mean rule (stable domain)
const M2_DEAD_COH:   f64 = 0.28;  // below → die (destructive)
const M2_MIN: usize = 2;
const M2_MAX: usize = 6;          // overcrowding → die

// ── Mode 3 ────────────────────────────────────────────────────────────────────
const M3_BETA:       f64 = 0.38;  // blend ratio: mean (1-β) + product (β)
const M3_BIRTH_COH:  f64 = 0.42;
const M3_SURV_COH:   f64 = 0.22;
const M3_MIN: usize = 2;
const M3_MAX: usize = 7;

type Grid = Vec<Vec<[f64; 4]>>;

fn make_grid() -> Grid { vec![vec![IDENTITY; COLS]; ROWS] }
fn is_alive(q: &[f64; 4]) -> bool { sigma(q) > 1e-9 }
fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

/// Normalized linear interpolation on S³ (NLERP — cheap, stable).
fn nlerp(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    // Flip b to the same hemisphere as a to take the short arc.
    let (bw, bx, by, bz) = if dot4(a, b) < 0.0 {
        (-b[0], -b[1], -b[2], -b[3])
    } else {
        (b[0], b[1], b[2], b[3])
    };
    let s = 1.0 - t;
    let r = [s*a[0]+t*bw, s*a[1]+t*bx, s*a[2]+t*by, s*a[3]+t*bz];
    let len = (r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+r[3]*r[3]).sqrt();
    if len < 1e-12 { return *a; }
    [r[0]/len, r[1]/len, r[2]/len, r[3]/len]
}

// ── Neighbor collection ───────────────────────────────────────────────────────

fn collect_neighbors(grid: &Grid, r: usize, c: usize) -> (Vec<[f64; 4]>, [f64; 4], f64) {
    let mut nbrs = Vec::with_capacity(8);
    let mut sum  = [0.0f64; 4];
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if dr == 0 && dc == 0 { continue; }
            let nr = ((r as i32 + dr).rem_euclid(ROWS as i32)) as usize;
            let nc = ((c as i32 + dc).rem_euclid(COLS as i32)) as usize;
            let q = grid[nr][nc];
            if is_alive(&q) {
                sum[0] += q[0]; sum[1] += q[1]; sum[2] += q[2]; sum[3] += q[3];
                nbrs.push(q);
            }
        }
    }
    let len = (sum[0]*sum[0]+sum[1]*sum[1]+sum[2]*sum[2]+sum[3]*sum[3]).sqrt();
    let coh = if nbrs.is_empty() { 0.0 } else { len / nbrs.len() as f64 };
    (nbrs, sum, coh)
}

fn normalize_sum(sum: &[f64; 4]) -> Option<[f64; 4]> {
    let len = (sum[0]*sum[0]+sum[1]*sum[1]+sum[2]*sum[2]+sum[3]*sum[3]).sqrt();
    if len < 1e-12 { None } else { Some([sum[0]/len, sum[1]/len, sum[2]/len, sum[3]/len]) }
}

/// The pair of alive neighbors with the smallest dot product (most different).
fn most_different_pair(nbrs: &[[f64; 4]]) -> Option<([f64; 4], [f64; 4])> {
    if nbrs.len() < 2 { return None; }
    let mut best = f64::MAX;
    let mut pair = None;
    for i in 0..nbrs.len() {
        for j in i+1..nbrs.len() {
            let d = dot4(&nbrs[i], &nbrs[j]);
            if d < best { best = d; pair = Some((nbrs[i], nbrs[j])); }
        }
    }
    pair
}

fn hamilton_all(nbrs: &[[f64; 4]]) -> [f64; 4] {
    nbrs.iter().fold(IDENTITY, |acc, q| compose(&acc, q))
}

// ── Step functions ────────────────────────────────────────────────────────────

fn step_spectrum(grid: &Grid) -> Grid {
    let mut next = make_grid();
    for r in 0..ROWS {
        for c in 0..COLS {
            let (nbrs, sum, coh) = collect_neighbors(grid, r, c);
            if nbrs.len() < M1_MIN { continue; }
            let alive = is_alive(&grid[r][c]);
            let thresh = if alive { M1_SURV_COH } else { M1_BIRTH_COH };
            if coh >= thresh {
                if let Some(m) = normalize_sum(&sum) { next[r][c] = m; }
            }
        }
    }
    next
}

fn step_resonance(grid: &Grid) -> Grid {
    let mut next = make_grid();
    for r in 0..ROWS {
        for c in 0..COLS {
            let (nbrs, sum, coh) = collect_neighbors(grid, r, c);
            let n = nbrs.len();
            if n < M2_MIN || n > M2_MAX { continue; }

            if coh >= M2_STABLE_COH {
                // Integration: stable domain, drift toward mean.
                if let Some(m) = normalize_sum(&sum) { next[r][c] = m; }
            } else if coh >= M2_DEAD_COH {
                // Creation: Hamilton product of the most different pair.
                // One clean resonance event produces one new frequency.
                if let Some((qa, qb)) = most_different_pair(&nbrs) {
                    let p = compose(&qa, &qb);
                    if sigma(&p) > 1e-9 {
                        next[r][c] = p;
                    } else if let Some(m) = normalize_sum(&sum) {
                        next[r][c] = m;
                    }
                }
            }
            // else: coh < M2_DEAD_COH → stays IDENTITY (die / dissolution)
        }
    }
    next
}

fn step_edge(grid: &Grid) -> Grid {
    let mut next = make_grid();
    for r in 0..ROWS {
        for c in 0..COLS {
            let (nbrs, sum, coh) = collect_neighbors(grid, r, c);
            let n = nbrs.len();
            if n < M3_MIN || n > M3_MAX { continue; }

            let alive = is_alive(&grid[r][c]);
            let thresh = if alive { M3_SURV_COH } else { M3_BIRTH_COH };
            if coh < thresh { continue; }

            let mean = match normalize_sum(&sum) { Some(m) => m, None => continue };
            let prod = hamilton_all(&nbrs);

            next[r][c] = if sigma(&prod) > 1e-9 {
                nlerp(&mean, &prod, M3_BETA)
            } else {
                mean
            };
        }
    }
    next
}

// ── Color ─────────────────────────────────────────────────────────────────────

fn carrier_rgb(q: &[f64; 4]) -> Option<(u8, u8, u8)> {
    if !is_alive(q) { return None; }
    let (_, x, y, z) = (q[0], q[1], q[2], q[3]);
    let ax = (x*x + y*y + z*z).sqrt();
    let (nx, ny, nz) = if ax > 1e-9 { (x/ax, y/ax, z/ax) } else { return Some((210,210,210)); };
    let br = (sigma(q) / FRAC_PI_2).clamp(0.0, 1.0).sqrt();
    Some((
        ((nx + 1.0) * 127.5 * br) as u8,  // R = X = salience
        ((ny + 1.0) * 127.5 * br) as u8,  // G = Y = total      (known = G − B)
        ((nz + 1.0) * 127.5 * br) as u8,  // B = Z = unknown
    ))
}

fn color_block(q: &[f64; 4]) -> String {
    match carrier_rgb(q) {
        None => "  ".to_string(),
        Some((r, g, b)) => format!("\x1b[48;2;{r};{g};{b}m  \x1b[0m"),
    }
}

// ── Statistics ────────────────────────────────────────────────────────────────
//
// Discretise S² (the rotation-axis sphere) into N_THETA × N_PHI bins.
// For each alive cell, find which bin its rotation axis falls in.
// Compute Shannon entropy H = −Σ p·log₂(p) and count occupied bins.

// H_MAX: log₂(3800) — practical ceiling over the grid cell count.
const H_MAX_BITS: f64 = 11.892_789; // log2(3800)

struct Stats {
    alive:        usize,
    entropy_bits: f64,  // H in bits
    entropy_norm: f64,  // H / H_max  ∈ [0, 1]
}

fn compute_stats(grid: &Grid) -> Stats {
    use std::collections::HashMap;
    let mut rgb_counts: HashMap<(u8, u8, u8), u32> = HashMap::new();
    let mut alive = 0usize;

    for q in grid.iter().flat_map(|r| r.iter()) {
        if !is_alive(q) { continue; }
        alive += 1;
        if let Some(rgb) = carrier_rgb(q) {
            *rgb_counts.entry(rgb).or_insert(0) += 1;
        }
    }

    if alive == 0 { return Stats { alive: 0, entropy_bits: 0.0, entropy_norm: 0.0 }; }

    let n = alive as f64;
    let entropy_bits: f64 = rgb_counts.values()
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum();

    Stats {
        alive,
        entropy_bits,
        entropy_norm: (entropy_bits / H_MAX_BITS).clamp(0.0, 1.0),
    }
}

// ── Seeding ───────────────────────────────────────────────────────────────────

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * PI * u2;
    (r * t.cos(), r * t.sin())
}
fn uniform_s3(seed: &mut u64) -> [f64; 4] {
    let (w, x) = box_muller(lcg(seed).max(1e-300), lcg(seed));
    let (y, z) = box_muller(lcg(seed).max(1e-300), lcg(seed));
    let len = (w*w + x*x + y*y + z*z).sqrt();
    [w/len, x/len, y/len, z/len]
}

/// Mode 1: 15 random-carrier rectangular patches.
fn seed_spectrum(run_seed: u64) -> Grid {
    let mut seed = run_seed;
    let mut grid = make_grid();
    const PC: usize = 5; const PR: usize = 3;
    let quats: Vec<[f64; 4]> = (0..PR*PC).map(|_| uniform_s3(&mut seed)).collect();
    let ph = ROWS / PR; let pw = COLS / PC;
    for r in 0..ROWS { for c in 0..COLS {
        if lcg(&mut seed) >= 0.70 { continue; }
        let pr = (r/ph).min(PR-1); let pc = (c/pw).min(PC-1);
        grid[r][c] = quats[pr*PC+pc];
    }}
    grid
}

/// Modes 2 & 3: N equatorial sectors (σ = π/2, W = 0) at 2π/N intervals.
/// A random angular offset rotates the whole wheel each reseed.
fn seed_sectors(n: usize, density: f64, run_seed: u64) -> Grid {
    let mut seed = run_seed;
    let mut grid = make_grid();
    let offset = lcg(&mut seed) * 2.0 * PI;
    let primaries: Vec<[f64; 4]> = (0..n).map(|i| {
        let t = offset + (i as f64) * 2.0 * PI / n as f64;
        [0.0, t.cos(), t.sin(), 0.0]
    }).collect();
    let cr = ROWS as f64 / 2.0;
    let cc = COLS as f64 / 2.0;
    for r in 0..ROWS { for c in 0..COLS {
        if lcg(&mut seed) >= density { continue; }
        let angle = (r as f64 - cr).atan2(c as f64 - cc).rem_euclid(2.0 * PI);
        let sector = (angle / (2.0 * PI / n as f64)) as usize % n;
        grid[r][c] = primaries[sector];
    }}
    grid
}

// ── Render ────────────────────────────────────────────────────────────────────

fn entropy_bar(norm: f64, width: usize) -> String {
    let filled = (norm * width as f64).round() as usize;
    let empty  = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn sparkline(history: &[f64]) -> String {
    // history values are entropy_norm ∈ [0,1].
    // auto-scale to the observed range so early dynamics are visible.
    let lo = history.iter().cloned().fold(f64::MAX, f64::min);
    let hi = history.iter().cloned().fold(f64::MIN, f64::max);
    let span = (hi - lo).max(1e-6);
    let chars: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    history.iter().map(|&v| {
        let idx = (((v - lo) / span) * 8.0).round() as usize;
        chars[idx.min(8)]
    }).collect()
}

fn render(grid: &Grid, gen: usize, mode: u8, seed: u64,
          stats: &Stats, history: &[f64]) -> String {
    let dim = "\x1b[2m"; let rst = "\x1b[0m"; let bold = "\x1b[1m";
    let label = match mode {
        1 => "Spectrum  — coherence / integration",
        2 => "Resonance — pair Hamilton product at boundaries",
        _ => "Edge      — β=0.38 blend, neither side wins",
    };
    let mut out = String::with_capacity((ROWS + 4) * COLS * 24);

    // ── header ────────────────────────────────────────────────────────────────
    out.push_str(&format!(
        "  {bold}{label}{rst}  {dim}gen {gen:>5}  alive {:>5}  \
         1=spectrum  2=resonance  3=edge  SPACE=reseed  Q=quit{rst}\n",
        stats.alive
    ));

    // ── stats line ────────────────────────────────────────────────────────────
    // Shannon entropy bar (20 chars) + value in bits + distinct color count.
    let bar = entropy_bar(stats.entropy_norm, 20);
    out.push_str(&format!(
        "  {dim}H {bar} {:.2}/{:.2} bits{rst}\n",
        stats.entropy_bits, H_MAX_BITS,
    ));

    // ── grid ──────────────────────────────────────────────────────────────────
    for row in grid.iter() {
        out.push_str("  ");
        for q in row.iter() { out.push_str(&color_block(q)); }
        out.push('\n');
    }

    // ── entropy sparkline ─────────────────────────────────────────────────────
    // Shows H_norm over the last ≤96 generations — you watch it rise or fall.
    if !history.is_empty() {
        out.push_str(&format!(
            "  {dim}H ╠{}╣  entropy history{rst}\n",
            sparkline(history)
        ));
    }

    // ── footer ────────────────────────────────────────────────────────────────
    out.push_str(&format!(
        "  {dim}R=salience  G=total  B=unknown  yellow=known(G−B)  brightness=σ  seed {:016x}{rst}\n",
        seed
    ));
    out
}

// ── Terminal raw mode ─────────────────────────────────────────────────────────

fn raw_on()  { let _ = std::process::Command::new("stty").args(["-F","/dev/tty","-echo","cbreak"]).status(); }
fn raw_off() { let _ = std::process::Command::new("stty").args(["-F","/dev/tty","echo","-cbreak"]).status(); }

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    use std::sync::mpsc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    print!("\x1b[?25l"); io::stdout().flush().unwrap();

    let (tx, rx) = mpsc::channel::<u8>();
    let running  = Arc::new(AtomicBool::new(true));
    let running2 = running.clone();
    std::thread::spawn(move || {
        use std::io::Read;
        raw_on();
        while running2.load(Ordering::Relaxed) {
            if let Ok(mut tty) = std::fs::File::open("/dev/tty") {
                let mut buf = [0u8; 1];
                if tty.read_exact(&mut buf).is_ok() { let _ = tx.send(buf[0]); }
            }
        }
    });

    let mut run_seed: u64 = 0xDEAD_BEEF_1337_4242;
    let mut mode: u8 = 1;
    let delay = Duration::from_millis(70);

    'outer: loop {
        let mut grid = match mode {
            1 => seed_spectrum(run_seed),
            2 => seed_sectors(6, 0.68, run_seed),
            _ => seed_sectors(2, 0.42, run_seed),
        };
        let mut gen = 0usize;
        // Entropy history for sparkline — keep last 96 values (fits in COLS width).
        let mut history: Vec<f64> = Vec::with_capacity(96);

        loop {
            while let Ok(k) = rx.try_recv() {
                match k {
                    b'q' | b'Q' => break 'outer,
                    b' ' => {
                        run_seed = run_seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                        gen = usize::MAX;
                    }
                    b'1' => { mode = 1; gen = usize::MAX; }
                    b'2' => { mode = 2; gen = usize::MAX; }
                    b'3' => { mode = 3; gen = usize::MAX; }
                    _ => {}
                }
            }
            if gen == usize::MAX { break; }

            let stats = compute_stats(&grid);
            // Push to history; keep at most 96 values.
            if history.len() >= 96 { history.remove(0); }
            history.push(stats.entropy_norm);

            print!("\x1b[H{}", render(&grid, gen, mode, run_seed, &stats, &history));
            io::stdout().flush().unwrap();

            if stats.alive == 0 {
                thread::sleep(Duration::from_millis(900));
                run_seed = run_seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                break;
            }

            grid = match mode {
                1 => step_spectrum(&grid),
                2 => step_resonance(&grid),
                _ => step_edge(&grid),
            };
            gen += 1;
            thread::sleep(delay);
        }
    }

    running.store(false, std::sync::atomic::Ordering::Relaxed);
    raw_off();
    print!("\x1b[?25h\x1b[2J\x1b[H");
    io::stdout().flush().unwrap();
}
