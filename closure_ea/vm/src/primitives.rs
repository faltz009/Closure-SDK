use closure_rs::hopf::decompose as hopf_decompose;
use closure_rs::groups::sphere::sphere_sigma as sigma;

/// ISA #4: DECOMPOSE. Hopf fibration → (σ, base[3], phase).
/// base = S² direction (WHAT kind of deviation).
/// phase = S¹ position (WHERE in the cycle).
pub fn decompose(q: &[f64; 4]) -> DecomposeResult {
    let s = sigma(q);
    let (base, phase) = hopf_decompose(q);
    DecomposeResult { sigma: s, base, phase }
}

#[derive(Debug, Clone, Copy)]
pub struct DecomposeResult {
    pub sigma: f64,
    pub base: [f64; 3],
    pub phase: f64,
}

/// Four outcomes. Death is not Halt.
#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    /// σ in (ε, π/2-ε): still computing.
    Continue(f64),
    /// σ < ε: success. Carries the closure element.
    Closure([f64; 4]),
    /// σ > π/2-ε: failure. Maximum departure from identity. σ = arccos(|w|) ∈ [0, π/2].
    Death([f64; 4]),
    /// Program exhausted without deciding.
    Halt([f64; 4]),
}
