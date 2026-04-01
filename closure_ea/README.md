# Closure EA

`closure_ea` is the umbrella computer module.

## Structure

- `closure_ea/dna/` — persistent geometric memory and database layer
- `closure_ea/vm/` — S³ virtual machine and execution layer
- `closure_ea/enkidu_alive/` — self-contained demo that stays active
- `closure_ea/archive/legacy_runtime/` — archived pre-computer EA runtime files

## Shared Rust Core

DNA and VM both use the shared Rust core in `rust/`.
The shared crate exports the low-level algebra plus the DNA table engine.

## Python Surface

- import DNA from `closure_ea.dna`
- run the DNA CLI with `python -m closure_ea.dna`

The old Trinity-era EA runtime has been archived so the top-level module now reflects the computer stack directly.
