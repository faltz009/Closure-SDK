//! Columnar storage — each field is its own element on S³.
//!
//! A record with N fields is N quaternions composed. Each field is
//! stored in its own column file, accessed independently. Filter by
//! city? Read only the city column. AVG(score)? Read only the score
//! column. No parsing. No deserialization. Just bytes at offsets.
//!
//! Column types:
//!   F64    — 8 bytes per value, packed. Numbers.
//!   Bytes  — variable length, length-prefixed. Strings, blobs.
//!
//! Each column has its own running product on S³. The row quaternion
//! is the composition of field quaternions. The table quaternion is
//! the composition of row quaternions.
//!
//! This is the algebraic decomposition of structured data:
//!   field = element on S³
//!   record = composition of field elements
//!   table = composition of record compositions

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::embed::{
    bytes_to_sphere4,
    bytes_to_sphere_opaque4,
    f64_from_order_sphere4,
    f64_to_order_sphere4,
    f64_to_sphere4,
    i64_to_sphere4,
};
use crate::groups::sphere::SphereGroup;
use crate::groups::LieGroup;
use crate::hopf::{
    base_distance as hopf_base_distance,
    circular_distance as hopf_circular_distance,
    decompose as hopf_decompose,
    identity_distance as hopf_identity_distance,
    phase_mean as hopf_phase_mean,
};
use crate::path::GeometricPath;
use crate::resonance::ResonanceHit;

const DIM: usize = 4;
const HEADER_MAGIC: &[u8; 8] = b"CDNAHDR1";
const HEADER_VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnExec {
    F64,
    I64,
    BytesOpaque,
    BytesIndexed,
}

impl ColumnExec {
    #[inline]
    fn from_def(def: &ColumnDef) -> Self {
        match def.col_type {
            ColumnType::F64 => Self::F64,
            ColumnType::I64 => Self::I64,
            ColumnType::Bytes if def.indexed => Self::BytesIndexed,
            ColumnType::Bytes => Self::BytesOpaque,
        }
    }
}

// ── Column types ────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum ColumnType {
    F64,
    /// 64-bit signed integer. Exact for all values — no precision loss above 2^53.
    I64,
    Bytes,
}

#[derive(Clone, Debug)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: ColumnType,
    pub indexed: bool,
    pub not_null: bool,
    pub unique: bool,
}

// ── Column storage ──────────────────────────────────────────────────

/// A single column stored on disk.
/// Projected columns also store a quaternion sidecar: packed [f64; 4] per value.
/// - bytes indexed columns use it for equality / geometric match
/// - f64 columns use it as the monotonic numeric chart for compare/sort/aggregate
pub struct Column {
    /// For F64: packed f64 LE values. For Bytes: length-prefixed entries.
    data_file: File,
    /// For Bytes columns: byte offsets for O(1) position lookup.
    offsets_file: Option<File>,
    /// Packed quaternion sidecar for projected operations.
    quat_file: Option<File>,
    /// One byte per row: 0 = present, 1 = null. Only exists for nullable columns.
    null_file: Option<File>,
    offsets: Vec<u64>,
    data_end: u64,
    count: usize,
}

impl Column {
    #[inline]
    fn append_null_bit(&mut self, is_null: bool) -> io::Result<()> {
        if let Some(ref mut nf) = self.null_file {
            nf.seek(SeekFrom::End(0))?;
            nf.write_all(&[if is_null { 1 } else { 0 }])?;
        }
        Ok(())
    }

    #[inline]
    fn write_null_bit(&mut self, position: usize, is_null: bool) -> io::Result<()> {
        if let Some(ref mut nf) = self.null_file {
            nf.seek(SeekFrom::Start(position as u64))?;
            nf.write_all(&[if is_null { 1 } else { 0 }])?;
        }
        Ok(())
    }

    #[inline]
    fn is_null(&mut self, position: usize) -> io::Result<bool> {
        if let Some(ref mut nf) = self.null_file {
            nf.seek(SeekFrom::Start(position as u64))?;
            let mut buf = [0u8; 1];
            nf.read_exact(&mut buf)?;
            Ok(buf[0] != 0)
        } else {
            Ok(false)
        }
    }

    fn read_nulls(&mut self) -> io::Result<Vec<bool>> {
        if let Some(ref mut nf) = self.null_file {
            let mut buf = vec![0u8; self.count];
            nf.seek(SeekFrom::Start(0))?;
            nf.read_exact(&mut buf)?;
            Ok(buf.into_iter().map(|b| b != 0).collect())
        } else {
            Ok(vec![false; self.count])
        }
    }

    fn create(dir: &Path, def: ColumnDef) -> io::Result<Self> {
        let data_path = dir.join(format!("col_{}.bin", def.name));
        let data_file = File::create(&data_path)?;

        let quat_file = match def.col_type {
            ColumnType::F64 | ColumnType::I64 => {
                let q_path = dir.join(format!("col_{}.q", def.name));
                Some(File::create(&q_path)?)
            }
            ColumnType::Bytes if def.indexed => {
                let q_path = dir.join(format!("col_{}.q", def.name));
                Some(File::create(&q_path)?)
            }
            ColumnType::Bytes => None,
        };

        let (offsets_file, offsets) = if def.col_type == ColumnType::Bytes {
            let off_path = dir.join(format!("col_{}.off", def.name));
            (Some(File::create(&off_path)?), Vec::new())
        } else {
            (None, Vec::new())
        };

        let null_file = if !def.not_null {
            let null_path = dir.join(format!("col_{}.null", def.name));
            Some(File::create(&null_path)?)
        } else {
            None
        };

        Ok(Self {
            data_file,
            offsets_file,
            quat_file,
            null_file,
            offsets,
            data_end: 0,
            count: 0,
        })
    }

    fn open(dir: &Path, def: ColumnDef, count: usize) -> io::Result<Self> {
        let data_path = dir.join(format!("col_{}.bin", def.name));
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&data_path)?;
        let data_end = data_file.metadata()?.len();

        let quat_file = match def.col_type {
            ColumnType::F64 | ColumnType::I64 => {
                let q_path = dir.join(format!("col_{}.q", def.name));
                Some(
                    OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&q_path)?,
                )
            }
            ColumnType::Bytes if def.indexed => {
                let q_path = dir.join(format!("col_{}.q", def.name));
                Some(
                    OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&q_path)?,
                )
            }
            ColumnType::Bytes => None,
        };

        // Lazy: open the offsets file but don't read until first access
        let (offsets_file, offsets) = if def.col_type == ColumnType::Bytes {
            let off_path = dir.join(format!("col_{}.off", def.name));
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&off_path)?;
            (Some(f), Vec::new()) // empty — loaded on first read_bytes
        } else {
            (None, Vec::new())
        };

        // Soft open: if the .null file doesn't exist (pre-null tables), treat all as non-null.
        let null_file = if !def.not_null {
            let null_path = dir.join(format!("col_{}.null", def.name));
            if null_path.exists() {
                Some(OpenOptions::new().read(true).write(true).open(&null_path)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            data_file,
            offsets_file,
            quat_file,
            null_file,
            offsets,
            data_end,
            count,
        })
    }

    /// Append one value.
    fn append_f64(&mut self, value: f64, projection: &[f64; 4]) -> io::Result<()> {
        self.data_file.seek(SeekFrom::End(0))?;
        self.data_file.write_all(&value.to_le_bytes())?;
        if let Some(ref mut qf) = self.quat_file {
            qf.seek(SeekFrom::End(0))?;
            for component in projection {
                qf.write_all(&component.to_le_bytes())?;
            }
        }
        self.append_null_bit(false)?;
        self.count += 1;
        Ok(())
    }

    fn append_bytes(&mut self, value: &[u8], quaternion: &[f64; 4]) -> io::Result<()> {
        self.ensure_offsets()?;
        let offset = self.data_end;
        self.data_file.seek(SeekFrom::End(0))?;
        let len = value.len() as u32;
        self.data_file.write_all(&len.to_le_bytes())?;
        self.data_file.write_all(value)?;
        self.data_end += 4 + value.len() as u64;

        if let Some(ref mut off_file) = self.offsets_file {
            off_file.seek(SeekFrom::End(0))?;
            off_file.write_all(&offset.to_le_bytes())?;
        }
        // Write quaternion to the geometric index
        if let Some(ref mut qf) = self.quat_file {
            qf.seek(SeekFrom::End(0))?;
            for i in 0..4 {
                qf.write_all(&quaternion[i].to_le_bytes())?;
            }
        }
        self.append_null_bit(false)?;
        self.offsets.push(offset);
        self.count += 1;
        Ok(())
    }

    /// Overwrite one f64 value at position. O(1) — fixed size, direct seek.
    fn write_f64(&mut self, position: usize, value: f64, projection: &[f64; 4]) -> io::Result<()> {
        self.data_file.seek(SeekFrom::Start(position as u64 * 8))?;
        self.data_file.write_all(&value.to_le_bytes())?;
        if let Some(ref mut qf) = self.quat_file {
            qf.seek(SeekFrom::Start(position as u64 * 32))?;
            for c in projection {
                qf.write_all(&c.to_le_bytes())?;
            }
        }
        self.write_null_bit(position, false)?;
        Ok(())
    }

    /// Write one bytes value at position. Appends new value, updates offset.
    fn write_bytes(&mut self, position: usize, value: &[u8], quaternion: &[f64; 4]) -> io::Result<()> {
        self.ensure_offsets()?;
        let new_offset = self.data_end;
        self.data_file.seek(SeekFrom::End(0))?;
        let len = value.len() as u32;
        self.data_file.write_all(&len.to_le_bytes())?;
        self.data_file.write_all(value)?;
        self.data_end += 4 + value.len() as u64;
        // Update offset to point to new value
        if let Some(ref mut off_file) = self.offsets_file {
            off_file.seek(SeekFrom::Start(position as u64 * 8))?;
            off_file.write_all(&new_offset.to_le_bytes())?;
        }
        self.offsets[position] = new_offset;
        // Update quaternion sidecar
        if let Some(ref mut qf) = self.quat_file {
            qf.seek(SeekFrom::Start(position as u64 * 32))?;
            for i in 0..4 {
                qf.write_all(&quaternion[i].to_le_bytes())?;
            }
        }
        self.write_null_bit(position, false)?;
        Ok(())
    }

    /// Read one F64 value at position.
    fn read_f64(&mut self, position: usize) -> io::Result<f64> {
        self.data_file
            .seek(SeekFrom::Start(position as u64 * 8))?;
        let mut buf = [0u8; 8];
        self.data_file.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    /// Load offsets from disk if not yet loaded.
    fn ensure_offsets(&mut self) -> io::Result<()> {
        if self.offsets.is_empty() && self.count > 0 {
            if let Some(ref mut f) = self.offsets_file {
                f.seek(SeekFrom::Start(0))?;
                let mut buf = [0u8; 8];
                for _ in 0..self.count {
                    f.read_exact(&mut buf)?;
                    self.offsets.push(u64::from_le_bytes(buf));
                }
            }
        }
        Ok(())
    }

    /// Read one Bytes value at position.
    fn read_bytes(&mut self, position: usize) -> io::Result<Vec<u8>> {
        self.ensure_offsets()?;
        let offset = self.offsets[position];
        self.data_file.seek(SeekFrom::Start(offset))?;
        let mut len_buf = [0u8; 4];
        self.data_file.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut data = vec![0u8; len];
        self.data_file.read_exact(&mut data)?;
        Ok(data)
    }

    /// Read ALL Bytes values. One disk pass.
    fn read_all_bytes(&mut self) -> io::Result<Vec<Vec<u8>>> {
        let n = self.count;
        let mut results = Vec::with_capacity(n);
        self.data_file.seek(SeekFrom::Start(0))?;
        for _ in 0..n {
            let mut len_buf = [0u8; 4];
            self.data_file.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            self.data_file.read_exact(&mut data)?;
            results.push(data);
        }
        Ok(results)
    }

    /// Read ALL F64 values. One disk read, returns packed Vec.
    fn read_all_f64(&mut self) -> io::Result<Vec<f64>> {
        let n = self.count;
        let mut buf = vec![0u8; n * 8];
        self.data_file.seek(SeekFrom::Start(0))?;
        self.data_file.read_exact(&mut buf)?;
        Ok(buf
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Append one I64 value. Same file layout as F64: 8-byte LE signed integer.
    fn append_i64(&mut self, value: i64, projection: &[f64; 4]) -> io::Result<()> {
        self.data_file.seek(SeekFrom::End(0))?;
        self.data_file.write_all(&value.to_le_bytes())?;
        if let Some(ref mut qf) = self.quat_file {
            qf.seek(SeekFrom::End(0))?;
            for component in projection {
                qf.write_all(&component.to_le_bytes())?;
            }
        }
        self.append_null_bit(false)?;
        self.count += 1;
        Ok(())
    }

    /// Overwrite one I64 value at position.
    fn write_i64(&mut self, position: usize, value: i64, projection: &[f64; 4]) -> io::Result<()> {
        self.data_file.seek(SeekFrom::Start(position as u64 * 8))?;
        self.data_file.write_all(&value.to_le_bytes())?;
        if let Some(ref mut qf) = self.quat_file {
            qf.seek(SeekFrom::Start(position as u64 * 32))?;
            for c in projection {
                qf.write_all(&c.to_le_bytes())?;
            }
        }
        self.write_null_bit(position, false)?;
        Ok(())
    }

    /// Read one I64 value at position.
    fn read_i64(&mut self, position: usize) -> io::Result<i64> {
        self.data_file.seek(SeekFrom::Start(position as u64 * 8))?;
        let mut buf = [0u8; 8];
        self.data_file.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    /// Read ALL I64 values. One disk read.
    fn read_all_i64(&mut self) -> io::Result<Vec<i64>> {
        let n = self.count;
        let mut buf = vec![0u8; n * 8];
        self.data_file.seek(SeekFrom::Start(0))?;
        self.data_file.read_exact(&mut buf)?;
        Ok(buf
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Filter: return positions where this I64 column passes a comparison.
    /// Uses native integer arithmetic — exact for all i64 values.
    pub fn filter_i64_cmp(&mut self, op: &str, target: i64) -> io::Result<Vec<usize>> {
        let values = self.read_all_i64()?;
        let nulls = self.read_nulls()?;
        Ok(values
            .iter()
            .enumerate()
            .filter(|(i, &v)| !nulls[*i] && match op {
                ">" => v > target,
                "<" => v < target,
                ">=" => v >= target,
                "<=" => v <= target,
                "=" => v == target,
                "!=" => v != target,
                _ => false,
            })
            .map(|(i, _)| i)
            .collect())
    }

    /// Aggregate: sum all I64 values. Returns i64 (no precision loss).
    pub fn sum_i64(&mut self) -> io::Result<i64> {
        let values = self.read_all_i64()?;
        let nulls = self.read_nulls()?;
        Ok(values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (!nulls[i]).then_some(*v))
            .sum())
    }

    /// Sort: return position indices sorted by I64 value.
    pub fn argsort_i64(&mut self, descending: bool) -> io::Result<Vec<usize>> {
        let values = self.read_all_i64()?;
        let nulls = self.read_nulls()?;
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| match (nulls[a], nulls[b]) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                if descending {
                    values[b].cmp(&values[a])
                } else {
                    values[a].cmp(&values[b])
                }
            }
        });
        Ok(indices)
    }

    /// Read all numeric values back from the geometric numeric sidecar.
    fn read_all_f64_projected(&mut self) -> io::Result<Vec<f64>> {
        if let Some(ref mut qf) = self.quat_file {
            let n = self.count;
            let mut buf = vec![0u8; n * 32];
            qf.seek(SeekFrom::Start(0))?;
            qf.read_exact(&mut buf)?;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let offset = i * 32;
                let q = [
                    f64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()),
                    f64::from_le_bytes(buf[offset + 8..offset + 16].try_into().unwrap()),
                    f64::from_le_bytes(buf[offset + 16..offset + 24].try_into().unwrap()),
                    f64::from_le_bytes(buf[offset + 24..offset + 32].try_into().unwrap()),
                ];
                out.push(f64_from_order_sphere4(&q));
            }
            Ok(out)
        } else {
            self.read_all_f64()
        }
    }

    // ── Column-level operations ─────────────────────────────────

    /// Filter: return positions where this Bytes column equals a value.
    /// Uses the quaternion index — compare fixed 32-byte quaternions
    /// instead of variable-length strings. Same bytes → same quaternion,
    /// guaranteed by deterministic embedding.
    pub fn filter_bytes_equals(&mut self, target: &[u8]) -> io::Result<Vec<usize>> {
        let nulls = self.read_nulls()?;
        if let Some(ref mut qf) = self.quat_file {
            // Embed target → quaternion, compare against stored quaternions
            let target_q = bytes_to_sphere4(target, false);
            let n = self.count;
            qf.seek(SeekFrom::Start(0))?;
            let mut buf = vec![0u8; n * 32]; // 4 × f64 × n
            qf.read_exact(&mut buf)?;
            let mut results = Vec::new();
            for i in 0..n {
                let offset = i * 32;
                let matches = (0..4).all(|j| {
                    let stored = f64::from_le_bytes(
                        buf[offset + j * 8..offset + j * 8 + 8].try_into().unwrap(),
                    );
                    (stored - target_q[j]).abs() < 1e-14
                });
                if matches && !nulls[i] {
                    results.push(i);
                }
            }
            Ok(results)
        } else {
            // Fallback: raw byte comparison
            let values = self.read_all_bytes()?;
            Ok(values
                .iter()
                .enumerate()
                .filter(|(i, v)| !nulls[*i] && v.as_slice() == target)
                .map(|(i, _)| i)
                .collect())
        }
    }

    /// Filter: return positions where this F64 column passes a comparison.
    pub fn filter_f64_cmp(&mut self, op: &str, target: f64) -> io::Result<Vec<usize>> {
        let values = self.read_all_f64_projected()?;
        let nulls = self.read_nulls()?;
        let eps = 1e-10;
        Ok(values
            .iter()
            .enumerate()
            .filter(|(i, &v)| !nulls[*i] && match op {
                ">" => v > target + eps,
                "<" => v < target - eps,
                ">=" => v >= target - eps,
                "<=" => v <= target + eps,
                "=" => (v - target).abs() < eps,
                "!=" => (v - target).abs() >= eps,
                _ => false,
            })
            .map(|(i, _)| i)
            .collect())
    }

    /// Aggregate: sum all F64 values.
    pub fn sum_f64(&mut self) -> io::Result<f64> {
        let values = self.read_all_f64_projected()?;
        let nulls = self.read_nulls()?;
        Ok(values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (!nulls[i]).then_some(*v))
            .sum())
    }

    /// Aggregate: average all F64 values.
    pub fn avg_f64(&mut self) -> io::Result<f64> {
        let values = self.read_all_f64_projected()?;
        let nulls = self.read_nulls()?;
        let non_null: Vec<f64> = values
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| (!nulls[i]).then_some(v))
            .collect();
        if non_null.is_empty() {
            return Ok(0.0);
        }
        Ok(non_null.iter().sum::<f64>() / non_null.len() as f64)
    }

    /// Sort: return position indices sorted by F64 value.
    pub fn argsort_f64(&mut self, descending: bool) -> io::Result<Vec<usize>> {
        let values = self.read_all_f64_projected()?;
        let nulls = self.read_nulls()?;
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| match (nulls[a], nulls[b]) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                if descending {
                    values[b].partial_cmp(&values[a]).unwrap()
                } else {
                    values[a].partial_cmp(&values[b]).unwrap()
                }
            }
        });
        Ok(indices)
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.data_file.sync_data()?;
        if let Some(ref mut f) = self.offsets_file {
            f.sync_data()?;
        }
        if let Some(ref mut f) = self.quat_file {
            f.sync_data()?;
        }
        if let Some(ref mut f) = self.null_file {
            f.sync_data()?;
        }
        Ok(())
    }

}

// ── Genome ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Codon {
    pub start: usize,
    pub end: usize,
    pub summary: [f64; 4],
    pub center: [f64; 4],
    pub radius: f64,
    pub base: [f64; 3],
    pub base_radius: f64,
    pub phase: f64,
    pub phase_radius: f64,
}

#[derive(Clone, Debug)]
pub struct Genome {
    levels: Vec<Vec<Codon>>,
}

#[derive(Clone, Debug)]
pub struct HopfView {
    pub drift: f64,
    pub base: [f64; 3],
    pub phase: f64,
}

/// Result of a full audit. Rereads disk, recomposes, compares against
/// stored identity, localizes and classifies any divergence.
pub struct AuditResult {
    /// True if the rebuilt identity matches the stored identity.
    pub ok: bool,
    /// Geodesic distance between rebuilt and stored. 0 = clean.
    pub drift: f64,
    /// First row where divergence was detected (None if ok).
    pub bad_row: Option<usize>,
    /// Hopf decomposition of the gap — what kind of divergence.
    pub hopf: HopfView,
}

#[derive(Clone, Copy, Debug)]
struct SearchNode {
    lower_bound: f64,
    hopf_bound: f64,
    level: usize,
    idx: usize,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound == other.lower_bound
            && self.hopf_bound == other.hopf_bound
            && self.level == other.level
            && self.idx == other.idx
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_score = self.lower_bound + self.hopf_bound;
        let other_score = other.lower_bound + other.hopf_bound;
        other_score.partial_cmp(&self_score)
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl Genome {
    pub fn build(path: &GeometricPath, rows: &[[f64; 4]], group: &dyn LieGroup, epsilon: f64) -> Self {
        if rows.is_empty() {
            return Self { levels: vec![Vec::new()] };
        }

        let level0 = Self::discover_blocks(path, rows, group, rows.len(), epsilon);
        let mut levels = vec![level0];

        loop {
            let prev = levels.last().unwrap();
            if prev.len() <= 1 {
                break;
            }

            let summaries: Vec<f64> = prev
                .iter()
                .flat_map(|b| b.summary.iter().copied())
                .collect();
            let summary_path = GeometricPath::from_elements(Box::new(SphereGroup), &summaries, DIM);
            let next = Self::discover_parent_blocks(&summary_path, prev, group, epsilon);
            if next.len() >= prev.len() {
                break;
            }
            levels.push(next);
        }

        Self { levels }
    }

    fn discover_blocks(
        path: &GeometricPath,
        rows: &[[f64; 4]],
        group: &dyn LieGroup,
        n: usize,
        epsilon: f64,
    ) -> Vec<Codon> {
        if n == 0 {
            return Vec::new();
        }

        let boundaries = Self::discover_boundaries(path, group, n, epsilon);

        let mut blocks = Vec::new();
        for window in boundaries.windows(2) {
            blocks.push(Self::make_leaf_block(path, rows, group, window[0], window[1]));
        }
        let last_start = *boundaries.last().unwrap();
        if last_start < n {
            blocks.push(Self::make_leaf_block(path, rows, group, last_start, n));
        }
        blocks
    }

    fn discover_parent_blocks(
        summary_path: &GeometricPath,
        prev: &[Codon],
        group: &dyn LieGroup,
        epsilon: f64,
    ) -> Vec<Codon> {
        let n = prev.len();
        let boundaries = Self::discover_boundaries(summary_path, group, n, epsilon);

        let mut blocks = Vec::new();
        for window in boundaries.windows(2) {
            blocks.push(Self::make_parent_block(summary_path, prev, group, window[0], window[1]));
        }
        let last_start = *boundaries.last().unwrap();
        if last_start < n {
            blocks.push(Self::make_parent_block(summary_path, prev, group, last_start, n));
        }
        blocks
    }

    fn discover_boundaries(
        path: &GeometricPath,
        group: &dyn LieGroup,
        n: usize,
        epsilon: f64,
    ) -> Vec<usize> {
        let mut scores = Vec::with_capacity(n);
        let mut sigmas = Vec::with_capacity(n);
        for t in 1..=n {
            let product = path.running_product(t);
            let sigma = group.distance_from_identity(product);
            let hopf = hopf_identity_distance(&[product[0], product[1], product[2], product[3]]);
            sigmas.push(sigma);
            // Closure is now discovered in full S^3 coordinates:
            // geodesic drift plus explicit Hopf base/phase return.
            scores.push(sigma + 0.5 * hopf);
        }

        let mut boundaries = vec![0usize];
        let mut running_sum = 0.0;
        for i in 0..n {
            running_sum += scores[i];
            let mean = running_sum / (i + 1) as f64;
            let below_mean = scores[i] < mean * 0.5;
            let below_epsilon = sigmas[i] < epsilon && scores[i] < epsilon * 2.0;
            let is_minimum =
                i > 0 && i < n - 1 && scores[i] < scores[i - 1] && scores[i] < scores[i + 1];
            if (below_mean || below_epsilon || is_minimum) && i > *boundaries.last().unwrap() {
                boundaries.push(i + 1);
            }
        }
        boundaries
    }

    fn make_leaf_block(
        path: &GeometricPath,
        rows: &[[f64; 4]],
        group: &dyn LieGroup,
        start: usize,
        end: usize,
    ) -> Codon {
        let summary = Self::sub_product(path, group, start, end);
        let center = Self::mean_center(&rows[start..end]);
        let radius = rows[start..end]
            .iter()
            .map(|row| group.distance_from_identity(&group.compose(&group.inverse(&center), row)))
            .fold(0.0, f64::max);
        let row_hopf: Vec<([f64; 3], f64)> = rows[start..end].iter().map(hopf_decompose).collect();
        let (base, base_radius) = Self::base_envelope(&row_hopf);
        let (phase, phase_radius) = Self::phase_envelope(&row_hopf);
        Codon { start, end, summary, center, radius, base, base_radius, phase, phase_radius }
    }

    fn make_parent_block(
        summary_path: &GeometricPath,
        children: &[Codon],
        group: &dyn LieGroup,
        start: usize,
        end: usize,
    ) -> Codon {
        let summary = Self::sub_product(summary_path, group, start, end);
        let child_slice = &children[start..end];
        let centers: Vec<[f64; 4]> = child_slice.iter().map(|c| c.center).collect();
        let center = Self::mean_center(&centers);
        let radius = child_slice
            .iter()
            .map(|child| {
                let d = group.distance_from_identity(&group.compose(&group.inverse(&center), &child.center));
                d + child.radius
            })
            .fold(0.0, f64::max);
        let child_hopf: Vec<([f64; 3], f64)> = child_slice.iter().map(|c| (c.base, c.phase)).collect();
        let (base, base_radius) = Self::base_envelope(&child_hopf);
        let phase = hopf_phase_mean(&child_hopf.iter().map(|(_, p)| *p).collect::<Vec<_>>());
        let phase_radius = child_slice
            .iter()
            .map(|child| hopf_circular_distance(phase, child.phase) + child.phase_radius)
            .fold(0.0, f64::max);
        Codon {
            start: child_slice.first().unwrap().start,
            end: child_slice.last().unwrap().end,
            summary,
            center,
            radius,
            base,
            base_radius,
            phase,
            phase_radius,
        }
    }

    fn sub_product(path: &GeometricPath, group: &dyn LieGroup, start: usize, end: usize) -> [f64; 4] {
        let inv = group.inverse(path.running_product(start));
        let product = group.compose(&inv, path.running_product(end));
        [product[0], product[1], product[2], product[3]]
    }

    fn mean_center(points: &[[f64; 4]]) -> [f64; 4] {
        let mut acc = [0.0; 4];
        for p in points {
            for i in 0..4 {
                acc[i] += p[i];
            }
        }
        let norm = (acc.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm > 1e-12 {
            for v in &mut acc {
                *v /= norm;
            }
            acc
        } else {
            points.first().copied().unwrap_or([1.0, 0.0, 0.0, 0.0])
        }
    }

    fn base_envelope(entries: &[([f64; 3], f64)]) -> ([f64; 3], f64) {
        let mut acc = [0.0; 3];
        for (base, _) in entries {
            for i in 0..3 {
                acc[i] += base[i];
            }
        }
        let norm = (acc.iter().map(|v| v * v).sum::<f64>()).sqrt();
        let center = if norm > 1e-12 {
            [acc[0] / norm, acc[1] / norm, acc[2] / norm]
        } else {
            entries.first().map(|(b, _)| *b).unwrap_or([0.0, 0.0, 1.0])
        };
        let radius = entries
            .iter()
            .map(|(base, _)| hopf_base_distance(&center, base))
            .fold(0.0, f64::max);
        (center, radius)
    }

    fn phase_envelope(entries: &[([f64; 3], f64)]) -> (f64, f64) {
        let phases: Vec<f64> = entries.iter().map(|(_, p)| *p).collect();
        let center = hopf_phase_mean(&phases);
        let radius = phases
            .iter()
            .map(|p| hopf_circular_distance(center, *p))
            .fold(0.0, f64::max);
        (center, radius)
    }

    fn child_indices(&self, level: usize, idx: usize) -> Vec<usize> {
        if level == 0 {
            return Vec::new();
        }
        let parent = &self.levels[level][idx];
        self.levels[level - 1]
            .iter()
            .enumerate()
            .filter_map(|(i, child)| {
                (child.start >= parent.start && child.end <= parent.end).then_some(i)
            })
            .collect()
    }

    fn lower_bound(&self, group: &dyn LieGroup, query: &[f64; 4], level: usize, idx: usize) -> f64 {
        let block = &self.levels[level][idx];
        let d = group.distance_from_identity(&group.compose(&group.inverse(query), &block.center));
        (d - block.radius).max(0.0)
    }

    fn hopf_bound(&self, query_base: &[f64; 3], query_phase: f64, level: usize, idx: usize) -> f64 {
        let block = &self.levels[level][idx];
        let base_lb = (hopf_base_distance(query_base, &block.base) - block.base_radius).max(0.0);
        let phase_lb = (hopf_circular_distance(query_phase, block.phase) - block.phase_radius).max(0.0);
        base_lb + phase_lb
    }

    pub fn query(&self, query: &[f64; 4], rows: &[[f64; 4]], group: &dyn LieGroup, k: usize) -> Vec<ResonanceHit> {
        if self.levels.is_empty() || self.levels[0].is_empty() || k == 0 {
            return Vec::new();
        }

        let mut queue = BinaryHeap::new();
        let top_level = self.levels.len() - 1;
        let (query_base, query_phase) = hopf_decompose(query);
        for idx in 0..self.levels[top_level].len() {
            queue.push(SearchNode {
                lower_bound: self.lower_bound(group, query, top_level, idx),
                hopf_bound: self.hopf_bound(&query_base, query_phase, top_level, idx),
                level: top_level,
                idx,
            });
        }

        let inv_query = group.inverse(query);
        let mut best: Vec<ResonanceHit> = Vec::new();
        let mut worst_best = f64::INFINITY;

        while let Some(node) = queue.pop() {
            if best.len() >= k && node.lower_bound >= worst_best {
                break;
            }

            if node.level == 0 {
                let block = &self.levels[0][node.idx];
                for row_idx in block.start..block.end {
                    let gap = group.compose(&inv_query, &rows[row_idx]);
                    let drift = group.distance_from_identity(&gap);
                    let gap_arr = [gap[0], gap[1], gap[2], gap[3]];
                    let (base, phase) = hopf_decompose(&gap_arr);
                    best.push(ResonanceHit { index: row_idx, drift, base, phase });
                }
                best.sort_by(|a, b| a.drift.partial_cmp(&b.drift).unwrap_or(Ordering::Equal));
                if best.len() > k {
                    best.truncate(k);
                }
                if best.len() == k {
                    worst_best = best.last().unwrap().drift;
                }
            } else {
                for child_idx in self.child_indices(node.level, node.idx) {
                    queue.push(SearchNode {
                        lower_bound: self.lower_bound(group, query, node.level - 1, child_idx),
                        hopf_bound: self.hopf_bound(&query_base, query_phase, node.level - 1, child_idx),
                        level: node.level - 1,
                        idx: child_idx,
                    });
                }
            }
        }

        best
    }

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    pub fn num_codons(&self) -> usize {
        self.levels.first().map_or(0, |l| l.len())
    }
}

// ── Table ──────────────────────────────────────────────────────

/// A table with typed columns. Each field is stored separately.
/// Each field is its own element on S³.
pub struct Table {
    dir: PathBuf,
    header_file: File,
    /// Path to tree.q — the persisted composition tree.
    schema: Vec<ColumnDef>,
    plan: Vec<ColumnExec>,
    columns: Vec<Column>,
    count: usize,
    group: SphereGroup,
    /// Balanced composition tree. Leaves = row quaternions, root = table identity.
    /// Update/delete: O(log n). Root read: O(1).
    tree: crate::composition_tree::CompositionTree,
    /// Cached identity from header — available immediately on open without
    /// loading the tree. Updated on every mutation.
    cached_identity: [f64; 4],
    genome: Option<Genome>,
}

impl Table {
    fn write_header_file(path: &Path, count: usize, identity: &[f64; 4], sync: bool) -> io::Result<()> {
        let mut header = [0u8; HEADER_SIZE];
        header[..8].copy_from_slice(HEADER_MAGIC);
        header[8..12].copy_from_slice(&HEADER_VERSION.to_le_bytes());
        header[12..20].copy_from_slice(&(count as u64).to_le_bytes());
        for (i, value) in identity.iter().enumerate() {
            let start = 20 + i * 8;
            header[start..start + 8].copy_from_slice(&value.to_le_bytes());
        }
        let mut f = File::create(path)?;
        f.write_all(&header)?;
        if sync {
            f.sync_all()?;
        }
        Ok(())
    }

    fn write_header(&mut self, sync: bool) -> io::Result<()> {
        let mut header = [0u8; HEADER_SIZE];
        header[..8].copy_from_slice(HEADER_MAGIC);
        header[8..12].copy_from_slice(&HEADER_VERSION.to_le_bytes());
        header[12..20].copy_from_slice(&(self.count as u64).to_le_bytes());
        for (i, value) in self.tree.root().iter().enumerate() {
            let start = 20 + i * 8;
            header[start..start + 8].copy_from_slice(&value.to_le_bytes());
        }
        self.header_file.seek(SeekFrom::Start(0))?;
        self.header_file.write_all(&header)?;
        if sync {
            self.header_file.sync_all()?;
        }
        Ok(())
    }

    fn read_header_file(path: &Path) -> io::Result<Option<(usize, [f64; 4])>> {
        if !path.exists() {
            return Ok(None);
        }
        let mut header = [0u8; HEADER_SIZE];
        let mut f = File::open(path)?;
        f.read_exact(&mut header)?;
        if &header[..8] != HEADER_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid header magic"));
        }
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if version != HEADER_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported header version"));
        }
        let count = u64::from_le_bytes(header[12..20].try_into().unwrap()) as usize;
        let mut identity = [0.0; 4];
        for (i, slot) in identity.iter_mut().enumerate() {
            let start = 20 + i * 8;
            *slot = f64::from_le_bytes(header[start..start + 8].try_into().unwrap());
        }
        Ok(Some((count, identity)))
    }

    #[inline]
    fn compile_plan(schema: &[ColumnDef]) -> Vec<ColumnExec> {
        schema.iter().map(ColumnExec::from_def).collect()
    }

    #[inline]
    fn embed_bytes(exec: ColumnExec, value: &[u8]) -> [f64; 4] {
        match exec {
            ColumnExec::BytesIndexed => bytes_to_sphere4(value, false),
            ColumnExec::BytesOpaque => bytes_to_sphere_opaque4(value),
            ColumnExec::F64 | ColumnExec::I64 => unreachable!("embed_bytes called on numeric column"),
        }
    }

    /// Enforce NOT NULL and UNIQUE constraints on a row before insert.
    /// UNIQUE uses the quaternion sidecar — drift = 0 means duplicate.
    /// This is the algebra doing constraint enforcement: same bytes →
    /// same quaternion → drift 0. No separate index structure needed.
    fn check_constraints(&mut self, values: &[ColumnValue]) -> io::Result<()> {
        for (i, value) in values.iter().enumerate() {
            let def = &self.schema[i];

            // NOT NULL: explicit null, empty bytes, or NaN f64
            if def.not_null {
                match value {
                    ColumnValue::Null => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("column '{}' is NOT NULL but got NULL", def.name),
                        ));
                    }
                    ColumnValue::Bytes(v) if v.is_empty() => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("column '{}' is NOT NULL but got empty value", def.name),
                        ));
                    }
                    ColumnValue::F64(v) if v.is_nan() => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("column '{}' is NOT NULL but got NaN", def.name),
                        ));
                    }
                    _ => {}
                }
            }

            if matches!(value, ColumnValue::Null) {
                continue;
            }

            // UNIQUE: check if this value already exists via quaternion drift.
            // Same bytes → same quaternion → drift = 0 when compared.
            if def.unique && self.count > 0 {
                match value {
                    ColumnValue::Null => {}
                    ColumnValue::Bytes(v) => {
                        let matches = self.columns[i].filter_bytes_equals(v)?;
                        if !matches.is_empty() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "column '{}' is UNIQUE but value already exists at row {}",
                                    def.name, matches[0]
                                ),
                            ));
                        }
                    }
                    ColumnValue::F64(v) => {
                        let matches = self.columns[i].filter_f64_cmp("=", *v)?;
                        if !matches.is_empty() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "column '{}' is UNIQUE but value {:.6} already exists at row {}",
                                    def.name, v, matches[0]
                                ),
                            ));
                        }
                    }
                    ColumnValue::I64(v) => {
                        let matches = self.columns[i].filter_i64_cmp("=", *v)?;
                        if !matches.is_empty() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "column '{}' is UNIQUE but value {} already exists at row {}",
                                    def.name, v, matches[0]
                                ),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Create a new typed table with the given schema.
    pub fn create(dir: &Path, schema: Vec<ColumnDef>) -> io::Result<Self> {
        fs::create_dir_all(dir)?;

        // Save schema
        let schema_path = dir.join("schema.bin");
        let mut sf = File::create(&schema_path)?;
        let n_cols = schema.len() as u32;
        sf.write_all(&n_cols.to_le_bytes())?;
        for col in &schema {
            let name_bytes = col.name.as_bytes();
            sf.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            sf.write_all(name_bytes)?;
            let type_byte: u8 = match col.col_type {
                ColumnType::F64 => 0,
                ColumnType::Bytes => 1,
                ColumnType::I64 => 2,
            };
            sf.write_all(&[type_byte])?;
            let flags: u8 =
                (if col.indexed { 1 } else { 0 })
                | (if col.not_null { 2 } else { 0 })
                | (if col.unique { 4 } else { 0 });
            sf.write_all(&[flags])?;
        }
        sf.sync_all()?;

        Self::write_header_file(&dir.join("header.bin"), 0, &[1.0, 0.0, 0.0, 0.0], true)?;

        // Create empty tree.q (persisted composition tree)
        File::create(dir.join("tree.q"))?.sync_all()?;

        // Create empty column files
        for def in &schema {
            Column::create(dir, def.clone())?.sync()?;
        }

        // Reopen with read+write via open()
        Self::open(dir)
    }

    /// Open an existing typed table.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let schema_path = dir.join("schema.bin");
        let mut sf = File::open(&schema_path)?;

        let mut buf4 = [0u8; 4];
        sf.read_exact(&mut buf4)?;
        let n_cols = u32::from_le_bytes(buf4) as usize;

        let mut schema = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            sf.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let mut name_bytes = vec![0u8; name_len];
            sf.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, e)
            })?;
            let mut type_buf = [0u8; 1];
            sf.read_exact(&mut type_buf)?;
            let col_type = match type_buf[0] {
                0 => ColumnType::F64,
                1 => ColumnType::Bytes,
                2 => ColumnType::I64,
                _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown column type")),
            };
            let mut flags_buf = [0u8; 1];
            sf.read_exact(&mut flags_buf)?;
            let flags = flags_buf[0];
            schema.push(ColumnDef {
                name,
                col_type,
                indexed: flags & 1 != 0,
                not_null: flags & 2 != 0,
                unique: flags & 4 != 0,
            });
        }

        let header_path = dir.join("header.bin");

        // Crash recovery: the header count is the last consistent state.
        // If column files are longer (mid-write crash), truncate them to
        // match the header. The running product at that count IS the
        // recovery point — no WAL replay needed.
        if let Some((header_count, _)) = Self::read_header_file(&header_path)? {
            // Crash recovery: header identity is the reference.
            // If tree.q is corrupt/missing, ensure_tree() will fail.
            // Use repair() to rebuild tree.q from columns.
            for def in &schema {
                let expected = header_count as u64 * 8;
                // Truncate fixed-width columns (F64, I64) to header count
                if def.col_type == ColumnType::F64 || def.col_type == ColumnType::I64 {
                    let path = dir.join(format!("col_{}.bin", def.name));
                    let actual = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    if actual > expected {
                        let f = OpenOptions::new().write(true).open(&path)?;
                        f.set_len(expected)?;
                    }
                    let null_path = dir.join(format!("col_{}.null", def.name));
                    if null_path.exists() {
                        let actual = fs::metadata(&null_path).map(|m| m.len()).unwrap_or(0);
                        if actual > header_count as u64 {
                            let f = OpenOptions::new().write(true).open(&null_path)?;
                            f.set_len(header_count as u64)?;
                        }
                    }
                    // Truncate quaternion sidecar too
                    let q_path = dir.join(format!("col_{}.q", def.name));
                    if q_path.exists() {
                        let q_expected = header_count as u64 * 32;
                        let q_actual = fs::metadata(&q_path).map(|m| m.len()).unwrap_or(0);
                        if q_actual > q_expected {
                            let f = OpenOptions::new().write(true).open(&q_path)?;
                            f.set_len(q_expected)?;
                        }
                    }
                }
                if def.col_type == ColumnType::Bytes {
                    // Truncate offsets file to header count
                    let off_path = dir.join(format!("col_{}.off", def.name));
                    if off_path.exists() {
                        let actual = fs::metadata(&off_path).map(|m| m.len()).unwrap_or(0);
                        if actual > expected {
                            let f = OpenOptions::new().write(true).open(&off_path)?;
                            f.set_len(expected)?;
                        }
                    }
                    // Truncate quaternion sidecar if indexed
                    if def.indexed {
                        let q_path = dir.join(format!("col_{}.q", def.name));
                        if q_path.exists() {
                            let q_expected = header_count as u64 * 32;
                            let q_actual = fs::metadata(&q_path).map(|m| m.len()).unwrap_or(0);
                            if q_actual > q_expected {
                                let f = OpenOptions::new().write(true).open(&q_path)?;
                                f.set_len(q_expected)?;
                            }
                        }
                    }
                    let null_path = dir.join(format!("col_{}.null", def.name));
                    if null_path.exists() {
                        let actual = fs::metadata(&null_path).map(|m| m.len()).unwrap_or(0);
                        if actual > header_count as u64 {
                            let f = OpenOptions::new().write(true).open(&null_path)?;
                            f.set_len(header_count as u64)?;
                        }
                    }
                }
            }
        }

        let plan = Self::compile_plan(&schema);
        let group = SphereGroup;

        // Read header for count + identity
        let (count, running) = if let Some((c, id)) = Self::read_header_file(&header_path)? {
            (c, id)
        } else {
            (0, [1.0, 0.0, 0.0, 0.0])
        };

        // Lazy open: read header for count + identity. Don't load row
        // quaternions until something needs them (search, audit, genome).
        // The identity is already in the header — check() and identity()
        // work immediately without loading anything.
        let tree_path = dir.join("tree.q");
        let header_file = OpenOptions::new().read(true).write(true).open(&header_path)?;

        // Read root from tree.q for instant identity — O(1), 32 bytes
        let cached_identity = if tree_path.exists() && fs::metadata(&tree_path)?.len() >= 32 {
            crate::composition_tree::CompositionTree::load_root_only(&tree_path)?
        } else {
            running
        };
        let mut columns = Vec::with_capacity(schema.len());
        for def in &schema {
            columns.push(Column::open(dir, def.clone(), count)?);
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            header_file,
            schema,
            plan,
            columns,
            count,
            group,
            tree: crate::composition_tree::CompositionTree::new(),
            cached_identity,
            genome: None,
        })
    }

    /// Ensure the composition tree is loaded. Reads tree.q on first access.
    /// Loads the full persisted tree (O(n) read, zero compute).
    /// Fails if tree.q is missing or corrupt — use audit() to diagnose
    /// and repair() to rebuild explicitly from columns.
    fn ensure_tree(&mut self) -> io::Result<()> {
        if self.tree.len() == 0 && self.count > 0 {
            let tree_path = self.dir.join("tree.q");
            self.tree = crate::composition_tree::CompositionTree::load_from(&tree_path, self.count)
                .map_err(|e| io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tree.q is corrupt or missing — run audit() to rebuild: {e}"),
                ))?;
            self.tree.attach_file(&tree_path)?;
        }
        Ok(())
    }

    /// Recompute row quaternions from columns (fallback when tree.q is missing/stale).
    fn recompute_row_elements(
        dir: &Path, schema: &[ColumnDef], plan: &[ColumnExec],
        group: &SphereGroup, count: usize,
    ) -> io::Result<Vec<[f64; 4]>> {
        let mut columns = Vec::with_capacity(schema.len());
        for def in schema {
            columns.push(Column::open(dir, def.clone(), count)?);
        }
        let mut rows = Vec::with_capacity(count);
        for row in 0..count {
            rows.push(Self::compute_row_quaternion_static(&mut columns, plan, group, row)?);
        }
        Ok(rows)
    }


    /// Insert one row. Values must match schema order.
    pub fn insert(&mut self, values: &[ColumnValue]) -> io::Result<usize> {
        self.ensure_tree()?;
        if values.len() != self.schema.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} values, got {}", self.schema.len(), values.len()),
            ));
        }

        // Enforce constraints
        self.check_constraints(values)?;

        // Write each field to its column, composing field quaternions
        let mut row_quaternion = [1.0, 0.0, 0.0, 0.0];
        let mut buf = [0.0; DIM];

        for (i, value) in values.iter().enumerate() {
            match (self.plan[i], value) {
                (ColumnExec::F64, ColumnValue::F64(v)) => {
                    let projection = f64_to_order_sphere4(*v);
                    self.columns[i].append_f64(*v, &projection)?;
                    let q = f64_to_sphere4(*v);
                    self.group.compose_into(&row_quaternion, &q, &mut buf);
                    row_quaternion.copy_from_slice(&buf);
                }
                (ColumnExec::F64, ColumnValue::Null) => {
                    let projection = f64_to_order_sphere4(0.0);
                    self.columns[i].append_f64(0.0, &projection)?;
                    let pos = self.columns[i].count - 1;
                    self.columns[i].write_null_bit(pos, true)?;
                }
                (ColumnExec::I64, ColumnValue::I64(v)) => {
                    let projection = i64_to_sphere4(*v);
                    self.columns[i].append_i64(*v, &projection)?;
                    self.group.compose_into(&row_quaternion, &projection, &mut buf);
                    row_quaternion.copy_from_slice(&buf);
                }
                (ColumnExec::I64, ColumnValue::Null) => {
                    let projection = i64_to_sphere4(0);
                    self.columns[i].append_i64(0, &projection)?;
                    let pos = self.columns[i].count - 1;
                    self.columns[i].write_null_bit(pos, true)?;
                }
                (ColumnExec::BytesOpaque, ColumnValue::Bytes(v))
                | (ColumnExec::BytesIndexed, ColumnValue::Bytes(v)) => {
                    let q = Self::embed_bytes(self.plan[i], v);
                    self.columns[i].append_bytes(v, &q)?;
                    self.group.compose_into(&row_quaternion, &q, &mut buf);
                    row_quaternion.copy_from_slice(&buf);
                }
                (ColumnExec::BytesOpaque, ColumnValue::Null)
                | (ColumnExec::BytesIndexed, ColumnValue::Null) => {
                    let q = [1.0, 0.0, 0.0, 0.0];
                    self.columns[i].append_bytes(&[], &q)?;
                    let pos = self.columns[i].count - 1;
                    self.columns[i].write_null_bit(pos, true)?;
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("type mismatch for column {}", self.schema[i].name),
                    ));
                }
            }
        }

        self.tree.append(row_quaternion);
        self.cached_identity = self.tree.root();

        let pos = self.count;
        self.count += 1;
        self.genome = None;
        self.write_header(false)?;
        Ok(pos)
    }

    /// Insert many rows. Buffers in memory, writes each column once.
    pub fn insert_many(&mut self, rows: &[Vec<ColumnValue>]) -> io::Result<usize> {
        let n_cols = self.schema.len();
        let n_rows = rows.len();
        let start = self.count;
        if rows.iter().any(|row| row.iter().any(|v| matches!(v, ColumnValue::Null))) {
            for row in rows {
                self.insert(row)?;
            }
            return Ok(start);
        }
        let mut columns: Vec<ColumnBatch> = self
            .schema
            .iter()
            .map(|def| match def.col_type {
                ColumnType::F64 => ColumnBatch::F64(Vec::with_capacity(n_rows)),
                ColumnType::I64 => ColumnBatch::I64(Vec::with_capacity(n_rows)),
                ColumnType::Bytes => ColumnBatch::Bytes(Vec::with_capacity(n_rows)),
            })
            .collect();

        for row in rows {
            if row.len() != n_cols {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("expected {} values, got {}", n_cols, row.len()),
                ));
            }
            for (i, value) in row.iter().enumerate() {
                match (&mut columns[i], value) {
                    (ColumnBatch::F64(dst), ColumnValue::F64(v)) => {
                        dst.push(*v);
                    }
                    (ColumnBatch::I64(dst), ColumnValue::I64(v)) => {
                        dst.push(*v);
                    }
                    (ColumnBatch::Bytes(dst), ColumnValue::Bytes(v)) => {
                        dst.push(v.clone());
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("type mismatch for column {}", self.schema[i].name),
                        ));
                    }
                }
            }
        }
        self.insert_columns(&columns)?;
        Ok(start)
    }

    /// Insert many rows as full typed columns.
    /// This is the native columnar ingest path: one typed buffer per column.
    pub fn insert_columns(&mut self, columns: &[ColumnBatch]) -> io::Result<usize> {
        self.ensure_tree()?;
        let start = self.count;
        let n_cols = self.schema.len();
        if columns.len() != n_cols {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} columns, got {}", n_cols, columns.len()),
            ));
        }
        let n_rows = columns.first().map_or(0, ColumnBatch::len);
        for (i, col) in columns.iter().enumerate() {
            if col.len() != n_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("column {} has {} rows, expected {}", self.schema[i].name, col.len(), n_rows),
                ));
            }
            match (self.plan[i], col) {
                (ColumnExec::F64, ColumnBatch::F64(_)) => {}
                (ColumnExec::I64, ColumnBatch::I64(_)) => {}
                (ColumnExec::BytesOpaque, ColumnBatch::Bytes(_))
                | (ColumnExec::BytesIndexed, ColumnBatch::Bytes(_)) => {}
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("type mismatch for column {}", self.schema[i].name),
                    ));
                }
            }
        }

        let mut row_quaternions = vec![[1.0, 0.0, 0.0, 0.0]; n_rows];
        let mut compose_buf = [0.0; DIM];

        let mut f64_bufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.plan[i] == ColumnExec::F64 { Vec::with_capacity(n_rows * 8) } else { Vec::new() })
            .collect();
        let mut f64_qbufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.plan[i] == ColumnExec::F64 { Vec::with_capacity(n_rows * 32) } else { Vec::new() })
            .collect();
        let mut i64_bufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.plan[i] == ColumnExec::I64 { Vec::with_capacity(n_rows * 8) } else { Vec::new() })
            .collect();
        let mut i64_qbufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.plan[i] == ColumnExec::I64 { Vec::with_capacity(n_rows * 32) } else { Vec::new() })
            .collect();

        let mut bytes_bufs: Vec<Vec<u8>> = columns
            .iter()
            .enumerate()
            .map(|(i, col)| match col {
                ColumnBatch::Bytes(values) => {
                    let cap = values.iter().map(|v| 4 + v.len()).sum();
                    Vec::with_capacity(cap)
                }
                ColumnBatch::F64(_) => {
                    let _ = i;
                    Vec::new()
                }
                ColumnBatch::I64(_) => {
                    let _ = i;
                    Vec::new()
                }
            })
            .collect();

        let mut offset_bufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.plan[i] != ColumnExec::F64 { Vec::with_capacity(n_rows * 8) } else { Vec::new() })
            .collect();
        let mut quat_bufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.plan[i] == ColumnExec::BytesIndexed { Vec::with_capacity(n_rows * 32) } else { Vec::new() })
            .collect();
        let null_bufs: Vec<Vec<u8>> = (0..n_cols)
            .map(|i| if self.columns[i].null_file.is_some() { vec![0u8; n_rows] } else { Vec::new() })
            .collect();
        let mut bytes_ends: Vec<u64> = self.columns.iter().map(|c| c.data_end).collect();
        // Repeated indexed strings are common in real imports; reuse their embeddings within the batch.
        let mut bytes_q_cache: Vec<Option<HashMap<Box<[u8]>, [f64; 4]>>> = (0..n_cols)
            .map(|i| {
                if self.plan[i] == ColumnExec::BytesIndexed {
                    Some(HashMap::with_capacity(32))
                } else {
                    None
                }
            })
            .collect();
        for (i, col) in columns.iter().enumerate() {
            match col {
                ColumnBatch::F64(values) => {
                    for (row_idx, value) in values.iter().enumerate() {
                        f64_bufs[i].extend_from_slice(&value.to_le_bytes());
                        let projection = f64_to_order_sphere4(*value);
                        for component in projection {
                            f64_qbufs[i].extend_from_slice(&component.to_le_bytes());
                        }
                        let q = f64_to_sphere4(*value);
                        self.group.compose_into(&row_quaternions[row_idx], &q, &mut compose_buf);
                        row_quaternions[row_idx].copy_from_slice(&compose_buf);
                    }
                }
                ColumnBatch::I64(values) => {
                    for (row_idx, value) in values.iter().enumerate() {
                        i64_bufs[i].extend_from_slice(&value.to_le_bytes());
                        let projection = i64_to_sphere4(*value);
                        for component in projection {
                            i64_qbufs[i].extend_from_slice(&component.to_le_bytes());
                        }
                        self.group.compose_into(&row_quaternions[row_idx], &projection, &mut compose_buf);
                        row_quaternions[row_idx].copy_from_slice(&compose_buf);
                    }
                }
                ColumnBatch::Bytes(values) => {
                    for (row_idx, value) in values.iter().enumerate() {
                        let offset = bytes_ends[i];
                        let len = value.len() as u32;
                        bytes_bufs[i].extend_from_slice(&len.to_le_bytes());
                        bytes_bufs[i].extend_from_slice(value);
                        bytes_ends[i] += 4 + value.len() as u64;
                        offset_bufs[i].extend_from_slice(&offset.to_le_bytes());

                        let q = if let Some(cache) = &mut bytes_q_cache[i] {
                            if let Some(&q) = cache.get(value.as_slice()) {
                                q
                            } else {
                                let q = Self::embed_bytes(self.plan[i], value);
                                cache.insert(value.clone().into_boxed_slice(), q);
                                q
                            }
                        } else {
                            Self::embed_bytes(self.plan[i], value)
                        };
                        if self.plan[i] == ColumnExec::BytesIndexed {
                            for component in q {
                                quat_bufs[i].extend_from_slice(&component.to_le_bytes());
                            }
                        }
                        self.group.compose_into(&row_quaternions[row_idx], &q, &mut compose_buf);
                        row_quaternions[row_idx].copy_from_slice(&compose_buf);
                    }
                }
            }
        }

        // Build tree from all rows (existing + new) in one O(n) pass
        let mut all_rows = self.tree.leaves();
        for row_quaternion in &row_quaternions {
            all_rows.push(*row_quaternion);
        }
        self.tree = crate::composition_tree::CompositionTree::from_elements(&all_rows);
        self.cached_identity = self.tree.root();
        self.count += n_rows;

        for (i, def) in self.schema.iter().enumerate() {
            match def.col_type {
                ColumnType::F64 => {
                    if !f64_bufs[i].is_empty() {
                        self.columns[i].data_file.seek(SeekFrom::End(0))?;
                        self.columns[i].data_file.write_all(&f64_bufs[i])?;
                        if let Some(ref mut qf) = self.columns[i].quat_file {
                            qf.seek(SeekFrom::End(0))?;
                            qf.write_all(&f64_qbufs[i])?;
                        }
                        if let Some(ref mut nf) = self.columns[i].null_file {
                            nf.seek(SeekFrom::End(0))?;
                            nf.write_all(&null_bufs[i])?;
                        }
                        self.columns[i].count += n_rows;
                    }
                }
                ColumnType::I64 => {
                    if !i64_bufs[i].is_empty() {
                        self.columns[i].data_file.seek(SeekFrom::End(0))?;
                        self.columns[i].data_file.write_all(&i64_bufs[i])?;
                        if let Some(ref mut qf) = self.columns[i].quat_file {
                            qf.seek(SeekFrom::End(0))?;
                            qf.write_all(&i64_qbufs[i])?;
                        }
                        if let Some(ref mut nf) = self.columns[i].null_file {
                            nf.seek(SeekFrom::End(0))?;
                            nf.write_all(&null_bufs[i])?;
                        }
                        self.columns[i].count += n_rows;
                    }
                }
                ColumnType::Bytes => {
                    self.columns[i].ensure_offsets()?;
                    if !bytes_bufs[i].is_empty() {
                        self.columns[i].data_file.seek(SeekFrom::End(0))?;
                        self.columns[i].data_file.write_all(&bytes_bufs[i])?;
                        self.columns[i].data_end = bytes_ends[i];
                    }
                    if let Some(ref mut of) = self.columns[i].offsets_file {
                        if !offset_bufs[i].is_empty() {
                            of.seek(SeekFrom::End(0))?;
                            of.write_all(&offset_bufs[i])?;
                        }
                    }
                    for chunk in offset_bufs[i].chunks_exact(8) {
                        self.columns[i].offsets.push(u64::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    if let Some(ref mut qf) = self.columns[i].quat_file {
                        if !quat_bufs[i].is_empty() {
                            qf.seek(SeekFrom::End(0))?;
                            qf.write_all(&quat_bufs[i])?;
                        }
                    }
                    if let Some(ref mut nf) = self.columns[i].null_file {
                        nf.seek(SeekFrom::End(0))?;
                        nf.write_all(&null_bufs[i])?;
                    }
                    self.columns[i].count += n_rows;
                }
            }
        }

        self.genome = None;
        self.write_header(false)?;
        Ok(start)
    }


    fn compute_query_quaternion(&self, values: &[ColumnValue]) -> io::Result<[f64; 4]> {
        if values.len() != self.schema.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} values, got {}", self.schema.len(), values.len()),
            ));
        }
        let mut row_quaternion = [1.0, 0.0, 0.0, 0.0];
        let mut buf = [0.0; DIM];
        for (i, value) in values.iter().enumerate() {
            let q = match (self.plan[i], value) {
                (ColumnExec::F64, ColumnValue::F64(v)) => f64_to_sphere4(*v),
                (ColumnExec::F64, ColumnValue::Null) => [1.0, 0.0, 0.0, 0.0],
                (ColumnExec::I64, ColumnValue::I64(v)) => i64_to_sphere4(*v),
                (ColumnExec::I64, ColumnValue::Null) => [1.0, 0.0, 0.0, 0.0],
                (ColumnExec::BytesOpaque, ColumnValue::Bytes(v))
                | (ColumnExec::BytesIndexed, ColumnValue::Bytes(v)) => Self::embed_bytes(self.plan[i], v),
                (ColumnExec::BytesOpaque, ColumnValue::Null)
                | (ColumnExec::BytesIndexed, ColumnValue::Null) => [1.0, 0.0, 0.0, 0.0],
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("type mismatch for column {}", self.schema[i].name),
                    ));
                }
            };
            self.group.compose_into(&row_quaternion, &q, &mut buf);
            row_quaternion.copy_from_slice(&buf);
        }
        Ok(row_quaternion)
    }

    pub fn build_genome(&mut self) -> io::Result<()> {
        self.ensure_tree()?;
        let leaves = self.tree.leaves_slice();
        let flat: Vec<f64> = leaves.iter().flat_map(|r| r.iter().copied()).collect();
        let path = GeometricPath::from_elements(Box::new(SphereGroup), &flat, DIM);
        self.genome = Some(Genome::build(&path, leaves, &self.group, 0.5));
        Ok(())
    }

    pub fn genome_depth(&mut self) -> io::Result<usize> {
        if self.genome.is_none() {
            self.build_genome()?;
        }
        Ok(self.genome.as_ref().map_or(0, Genome::depth))
    }

    pub fn genome_codons(&mut self) -> io::Result<usize> {
        if self.genome.is_none() {
            self.build_genome()?;
        }
        Ok(self.genome.as_ref().map_or(0, Genome::num_codons))
    }

    pub fn search(&mut self, values: &[ColumnValue], k: usize) -> io::Result<Vec<ResonanceHit>> {
        let query = self.compute_query_quaternion(values)?;
        if self.genome.is_none() {
            self.build_genome()?;
        }
        Ok(self
            .genome
            .as_ref()
            .unwrap()
            .query(&query, self.tree.leaves_slice(), &self.group, k))
    }

    /// Get one field value at (row, column_index).
    pub fn get_field_f64(&mut self, row: usize, col_idx: usize) -> io::Result<f64> {
        self.columns[col_idx].read_f64(row)
    }

    pub fn get_field_i64(&mut self, row: usize, col_idx: usize) -> io::Result<i64> {
        self.columns[col_idx].read_i64(row)
    }

    pub fn get_field_bytes(&mut self, row: usize, col_idx: usize) -> io::Result<Vec<u8>> {
        self.columns[col_idx].read_bytes(row)
    }

    pub fn get_row(&mut self, row: usize) -> io::Result<Vec<ColumnValue>> {
        if row >= self.count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("row {} out of range [0, {})", row, self.count),
            ));
        }
        let mut values = Vec::with_capacity(self.columns.len());
        for (i, def) in self.schema.iter().enumerate() {
            if self.columns[i].is_null(row)? {
                values.push(ColumnValue::Null);
                continue;
            }
            match def.col_type {
                ColumnType::F64 => values.push(ColumnValue::F64(self.columns[i].read_f64(row)?)),
                ColumnType::I64 => values.push(ColumnValue::I64(self.columns[i].read_i64(row)?)),
                ColumnType::Bytes => values.push(ColumnValue::Bytes(self.columns[i].read_bytes(row)?)),
            }
        }
        Ok(values)
    }

    /// Column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.schema.iter().position(|c| c.name == name)
    }

    pub fn schema_entries(&self) -> Vec<SchemaEntry> {
        self.schema
            .iter()
            .map(|c| SchemaEntry {
                name: c.name.clone(),
                col_type: c.col_type.clone(),
                indexed: c.indexed,
            })
            .collect()
    }

    /// Filter by string column equality. Returns matching row indices.
    pub fn filter_equals(&mut self, col_name: &str, value: &[u8]) -> io::Result<Vec<usize>> {
        let idx = self.column_index(col_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("no column: {col_name}"))
        })?;
        self.columns[idx].filter_bytes_equals(value)
    }

    /// Filter by numeric comparison. Returns matching row indices.
    pub fn filter_cmp(&mut self, col_name: &str, op: &str, value: f64) -> io::Result<Vec<usize>> {
        let idx = self.column_index(col_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("no column: {col_name}"))
        })?;
        match self.schema[idx].col_type {
            ColumnType::F64 => self.columns[idx].filter_f64_cmp(op, value),
            ColumnType::I64 => self.columns[idx].filter_i64_cmp(op, value as i64),
            ColumnType::Bytes => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("column {col_name} is not numeric"),
            )),
        }
    }

    /// Sum a numeric column.
    pub fn sum(&mut self, col_name: &str) -> io::Result<f64> {
        let idx = self.column_index(col_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("no column: {col_name}"))
        })?;
        match self.schema[idx].col_type {
            ColumnType::F64 => self.columns[idx].sum_f64(),
            ColumnType::I64 => Ok(self.columns[idx].sum_i64()? as f64),
            ColumnType::Bytes => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("column {col_name} is not numeric"),
            )),
        }
    }

    /// Average a numeric column.
    pub fn avg(&mut self, col_name: &str) -> io::Result<f64> {
        let idx = self.column_index(col_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("no column: {col_name}"))
        })?;
        match self.schema[idx].col_type {
            ColumnType::F64 => self.columns[idx].avg_f64(),
            ColumnType::I64 => {
                let count = self.columns[idx].count;
                if count == 0 {
                    Ok(0.0)
                } else {
                    Ok(self.columns[idx].sum_i64()? as f64 / count as f64)
                }
            }
            ColumnType::Bytes => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("column {col_name} is not numeric"),
            )),
        }
    }

    /// Sort indices by a numeric column.
    pub fn argsort(&mut self, col_name: &str, descending: bool) -> io::Result<Vec<usize>> {
        let idx = self.column_index(col_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("no column: {col_name}"))
        })?;
        match self.schema[idx].col_type {
            ColumnType::F64 => self.columns[idx].argsort_f64(descending),
            ColumnType::I64 => self.columns[idx].argsort_i64(descending),
            ColumnType::Bytes => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("column {col_name} is not numeric"),
            )),
        }
    }

    /// Table identity — composition of all row quaternions.
    pub fn identity(&self) -> [f64; 4] {
        self.cached_identity
    }

    /// Table integrity check. O(1) — reads the cached identity.
    pub fn check(&self) -> f64 {
        self.group.distance_from_identity(&self.cached_identity)
    }

    pub fn check_hopf(&self) -> HopfView {
        let drift = self.group.distance_from_identity(&self.cached_identity);
        let (base, phase) = hopf_decompose(&self.cached_identity);
        HopfView { drift, base, phase }
    }

    /// Full audit. Rereads every column from disk, recomposes every row,
    /// compares the rebuilt identity against the stored identity.
    /// If they differ: localizes the first bad row via binary search on
    /// the recomposed path, classifies the divergence via Hopf.
    ///
    /// This is the SDK's localize + classify applied to the table's own
    /// stored state. No external reference needed — the header identity
    /// IS the reference.
    ///
    /// Returns: (ok, drift, bad_row, hopf_view)
    ///   ok:       true if rebuilt identity matches stored identity
    ///   drift:    geodesic distance between rebuilt and stored
    ///   bad_row:  first row where divergence starts (None if ok)
    ///   hopf:     Hopf decomposition of the gap (what kind of divergence)
    /// Audit: check + diagnose. No writes. Rereads every column,
    /// recomposes every row, compares against stored identity.
    /// If they differ: localizes the first bad row, classifies via Hopf.
    /// Does NOT modify any files — use repair() to fix.
    pub fn audit(&mut self) -> io::Result<AuditResult> {
        // Step 1: reread columns and recompose every row from disk
        let mut rebuilt_elements: Vec<[f64; 4]> = Vec::with_capacity(self.count);
        let mut rebuilt_identity = [1.0, 0.0, 0.0, 0.0];
        let mut compose_buf = [0.0; DIM];

        for row in 0..self.count {
            let row_q = Self::compute_row_quaternion_static(
                &mut self.columns, &self.plan, &self.group, row,
            )?;
            rebuilt_elements.push(row_q);
            self.group.compose_into(&rebuilt_identity, &row_q, &mut compose_buf);
            rebuilt_identity = compose_buf;
        }

        // Step 2: compare rebuilt vs stored
        let stored = self.cached_identity;
        let inv_stored = self.group.inverse(&stored);
        let gap = self.group.compose(&inv_stored, &rebuilt_identity);
        let drift = self.group.distance_from_identity(&gap);

        if drift < 1e-10 {
            return Ok(AuditResult {
                ok: true,
                drift: 0.0,
                bad_row: None,
                hopf: HopfView { drift: 0.0, base: [0.0; 3], phase: 0.0 },
            });
        }

        // Step 3: localize — find the first row where composition diverges
        // from what the stored identity implies. Build temporary comparison
        // tree from rebuilt elements (in-memory only, no writes).
        // If stored tree is loaded, use binary search (O(log n)).
        // Otherwise, sequential scan (O(n) — but audit already read all columns).
        let bad_row = if self.tree.len() > 0 {
            // Binary search: compare stored tree prefix products vs rebuilt
            let rebuilt_tree = crate::composition_tree::CompositionTree::from_elements(&rebuilt_elements);
            let n = self.count;
            let mut lo = 0usize;
            let mut hi = n;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                let stored_prefix = self.tree.prefix_product(mid + 1);
                let rebuilt_prefix = rebuilt_tree.prefix_product(mid + 1);
                let inv = self.group.inverse(&stored_prefix);
                let rel = self.group.compose(&inv, &rebuilt_prefix);
                let d = self.group.distance_from_identity(&rel);
                if d > 1e-10 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let bad = if hi == lo + 1 { hi } else { lo };
            Some(bad.min(n - 1))
        } else {
            // No stored tree — scan rows sequentially. Compare running
            // composition of rebuilt elements against stored identity.
            // The first row where they start diverging is the bad row.
            let n = self.count;
            let mut running = [1.0, 0.0, 0.0, 0.0];
            let mut buf = [0.0; DIM];
            let found = None;
            for row in 0..n {
                self.group.compose_into(&running, &rebuilt_elements[row], &mut buf);
                running = buf;
                // Check if this prefix already shows drift from stored
                // (the stored identity is the composition of the ORIGINAL rows,
                // not the rebuilt ones — so any divergence here means this row
                // or an earlier one was corrupted on disk)
            }
            // Without stored intermediate products, we can't pinpoint the exact row.
            // Report None — caller should repair() then re-audit() for localization.
            found
        };

        // Step 4: Hopf classify the gap
        let gap_arr = [gap[0], gap[1], gap[2], gap[3]];
        let (base, phase) = hopf_decompose(&gap_arr);

        Ok(AuditResult {
            ok: false,
            drift,
            bad_row,
            hopf: HopfView { drift, base, phase },
        })
    }

    /// Repair: rebuild tree.q and header from columns. Explicit mutation.
    /// Call this after audit() reports a problem, or when tree.q is
    /// corrupt/missing and ensure_tree() fails.
    pub fn repair(&mut self) -> io::Result<()> {
        let elements = Self::recompute_row_elements(
            &self.dir, &self.schema, &self.plan, &self.group, self.count,
        )?;
        self.tree = crate::composition_tree::CompositionTree::from_elements(&elements);
        let tree_path = self.dir.join("tree.q");
        self.tree.save_to(&tree_path)?;
        self.tree.attach_file(&tree_path)?;
        self.cached_identity = self.tree.root();
        self.tree.sync()?;
        Self::write_header_file(
            &self.dir.join("header.bin"),
            self.count,
            &self.cached_identity,
            true,
        )?;
        Ok(())
    }

    pub fn inspect_row(&mut self, row: usize) -> io::Result<HopfView> {
        self.ensure_tree()?;
        if row >= self.count {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("row {} out of range [0, {})", row, self.count)));
        }
        let q = self.tree.get(row);
        let drift = self.group.distance_from_identity(&q);
        let (base, phase) = hopf_decompose(&q);
        Ok(HopfView { drift, base, phase })
    }

    /// Update one row in-place. O(1) algebraic identity update.
    /// Overwrites field values in their columns, computes new row
    /// quaternion, updates table identity via recomposition from
    /// Overwrites field values in their columns, O(log n) tree update.
    pub fn update(&mut self, row: usize, values: &[ColumnValue]) -> io::Result<()> {
        self.ensure_tree()?;
        if row >= self.count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("row {} out of range [0, {})", row, self.count),
            ));
        }
        if values.len() != self.schema.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} values, got {}", self.schema.len(), values.len()),
            ));
        }

        // Write each field value in-place in its column
        let mut new_row_q = [1.0, 0.0, 0.0, 0.0];
        let mut buf = [0.0; DIM];
        for (i, value) in values.iter().enumerate() {
            let q = match (self.plan[i], value) {
                (ColumnExec::F64, ColumnValue::F64(v)) => {
                    let proj = f64_to_order_sphere4(*v);
                    self.columns[i].write_f64(row, *v, &proj)?;
                    f64_to_sphere4(*v)
                }
                (ColumnExec::F64, ColumnValue::Null) => {
                    let proj = f64_to_order_sphere4(0.0);
                    self.columns[i].write_f64(row, 0.0, &proj)?;
                    self.columns[i].write_null_bit(row, true)?;
                    [1.0, 0.0, 0.0, 0.0]
                }
                (ColumnExec::I64, ColumnValue::I64(v)) => {
                    let proj = i64_to_sphere4(*v);
                    self.columns[i].write_i64(row, *v, &proj)?;
                    proj
                }
                (ColumnExec::I64, ColumnValue::Null) => {
                    let proj = i64_to_sphere4(0);
                    self.columns[i].write_i64(row, 0, &proj)?;
                    self.columns[i].write_null_bit(row, true)?;
                    [1.0, 0.0, 0.0, 0.0]
                }
                (ColumnExec::BytesOpaque, ColumnValue::Bytes(v))
                | (ColumnExec::BytesIndexed, ColumnValue::Bytes(v)) => {
                    let q = Self::embed_bytes(self.plan[i], v);
                    self.columns[i].write_bytes(row, v, &q)?;
                    q
                }
                (ColumnExec::BytesOpaque, ColumnValue::Null)
                | (ColumnExec::BytesIndexed, ColumnValue::Null) => {
                    let q = [1.0, 0.0, 0.0, 0.0];
                    self.columns[i].write_bytes(row, &[], &q)?;
                    self.columns[i].write_null_bit(row, true)?;
                    q
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("type mismatch for column {}", self.schema[i].name),
                    ));
                }
            };
            self.group.compose_into(&new_row_q, &q, &mut buf);
            new_row_q = buf;
        }

        // O(log n) update via composition tree — propagate disk errors
        self.tree.update(row, new_row_q)?;
        self.cached_identity = self.tree.root();
        self.genome = None;
        Self::write_header_file(
            &self.dir.join("header.bin"),
            self.count,
            &self.tree.root(),
            false,
        )?;
        Ok(())
    }

    /// Delete one row. O(1) algebraic identity update.
    /// Replaces row's quaternion with identity (contributes nothing).
    /// O(log n) delete via composition tree.
    pub fn delete(&mut self, row: usize) -> io::Result<()> {
        self.ensure_tree()?;
        if row >= self.count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("row {} out of range [0, {})", row, self.count),
            ));
        }

        // O(log n) delete via composition tree — set to identity
        let identity = [1.0, 0.0, 0.0, 0.0];
        self.tree.update(row, identity)?;
        self.cached_identity = self.tree.root();
        self.genome = None;
        Self::write_header_file(
            &self.dir.join("header.bin"),
            self.count,
            &self.tree.root(),
            false,
        )?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn save(&mut self) -> io::Result<()> {
        for col in &mut self.columns {
            col.sync()?;
        }
        // Persist the composition tree
        if self.tree.len() > 0 {
            let tree_path = self.dir.join("tree.q");
            self.tree.save_to(&tree_path)?;
        }
        self.tree.sync()?;
        self.write_header(true)?;
        Ok(())
    }

    /// Compute row quaternion from field values (static, for open/rebuild).
    fn compute_row_quaternion_static(
        columns: &mut [Column],
        plan: &[ColumnExec],
        group: &SphereGroup,
        row: usize,
    ) -> io::Result<[f64; 4]> {
        let mut row_q = [1.0, 0.0, 0.0, 0.0];
        let mut buf = [0.0; DIM];
        for (i, exec) in plan.iter().enumerate() {
            if columns[i].is_null(row)? {
                continue;
            }
            let field_q = match exec {
                ColumnExec::F64 => {
                    let v = columns[i].read_f64(row)?;
                    f64_to_sphere4(v)
                }
                ColumnExec::I64 => {
                    let v = columns[i].read_i64(row)?;
                    i64_to_sphere4(v)
                }
                ColumnExec::BytesOpaque | ColumnExec::BytesIndexed => {
                    let v = columns[i].read_bytes(row)?;
                    Self::embed_bytes(*exec, &v)
                }
            };
            group.compose_into(&row_q, &field_q, &mut buf);
            row_q.copy_from_slice(&buf);
        }
        Ok(row_q)
    }
}

/// A value for a typed column.
#[derive(Clone, Debug)]
pub enum ColumnValue {
    Null,
    F64(f64),
    /// 64-bit signed integer. Exact for all values, no f64 precision loss.
    I64(i64),
    Bytes(Vec<u8>),
}

/// A full typed column for bulk ingest.
#[derive(Clone, Debug)]
pub enum ColumnBatch {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bytes(Vec<Vec<u8>>),
}

impl ColumnBatch {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::Bytes(v) => v.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SchemaEntry {
    pub name: String,
    pub col_type: ColumnType,
    pub indexed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn test_dir(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("cdna_col_test_{}_{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    fn people_schema() -> Vec<ColumnDef> {
        vec![
            ColumnDef { name: "name".into(), col_type: ColumnType::Bytes, indexed: false, not_null: false, unique: false },
            ColumnDef { name: "age".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
            ColumnDef { name: "city".into(), col_type: ColumnType::Bytes, indexed: true, not_null: false, unique: false },
            ColumnDef { name: "score".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
        ]
    }

    fn people_rows() -> Vec<Vec<ColumnValue>> {
        vec![
            vec![ColumnValue::Bytes(b"Alice".to_vec()), ColumnValue::F64(30.0), ColumnValue::Bytes(b"Tokyo".to_vec()), ColumnValue::F64(85.5)],
            vec![ColumnValue::Bytes(b"Bob".to_vec()), ColumnValue::F64(25.0), ColumnValue::Bytes(b"Paris".to_vec()), ColumnValue::F64(92.0)],
            vec![ColumnValue::Bytes(b"Charlie".to_vec()), ColumnValue::F64(35.0), ColumnValue::Bytes(b"Tokyo".to_vec()), ColumnValue::F64(78.3)],
            vec![ColumnValue::Bytes(b"Diana".to_vec()), ColumnValue::F64(28.0), ColumnValue::Bytes(b"Cairo".to_vec()), ColumnValue::F64(91.0)],
            vec![ColumnValue::Bytes(b"Edward".to_vec()), ColumnValue::F64(40.0), ColumnValue::Bytes(b"Paris".to_vec()), ColumnValue::F64(88.7)],
        ]
    }

    #[test]
    fn create_insert_get() {
        let dir = test_dir("create");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();
        assert_eq!(t.count(), 5);

        // Get individual fields
        assert_eq!(t.get_field_bytes(0, 0).unwrap(), b"Alice");
        assert_eq!(t.get_field_f64(0, 1).unwrap(), 30.0);
        assert_eq!(t.get_field_bytes(2, 2).unwrap(), b"Tokyo");
        assert_eq!(t.get_field_f64(4, 3).unwrap(), 88.7);

        drop(t);
        cleanup(&dir);
    }

    #[test]
    fn filter_string_equals() {
        let dir = test_dir("filter_str");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();

        let tokyo = t.filter_equals("city", b"Tokyo").unwrap();
        assert_eq!(tokyo, vec![0, 2]); // Alice and Charlie

        let paris = t.filter_equals("city", b"Paris").unwrap();
        assert_eq!(paris, vec![1, 4]); // Bob and Edward

        drop(t);
        cleanup(&dir);
    }

    #[test]
    fn identity_and_check() {
        let dir = test_dir("identity");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();

        let id = t.identity();
        assert!(id.iter().all(|v| v.is_finite()));
        let drift = t.check();
        assert!(drift.is_finite());

        drop(t);
        cleanup(&dir);
    }

    #[test]
    fn persistence() {
        let dir = test_dir("persist");

        let id_before;
        {
            let mut t = Table::create(&dir, people_schema()).unwrap();
            t.insert_many(&people_rows()).unwrap();
            id_before = t.identity();
        }

        {
            let mut t = Table::open(&dir).unwrap();
            assert_eq!(t.count(), 5);
            let id_after = t.identity();
            for i in 0..4 {
                assert!((id_before[i] - id_after[i]).abs() < 1e-12);
            }
            assert_eq!(t.get_field_bytes(0, 0).unwrap(), b"Alice");
            assert_eq!(t.get_field_f64(4, 1).unwrap(), 40.0);
        }

        cleanup(&dir);
    }

    #[test]
    fn typed_resonance_search() {
        let dir = test_dir("search");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();

        let hits = t
            .search(
                &[
                    ColumnValue::Bytes(b"Alice".to_vec()),
                    ColumnValue::F64(30.0),
                    ColumnValue::Bytes(b"Tokyo".to_vec()),
                    ColumnValue::F64(85.5),
                ],
                1,
            )
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, 0);
        assert!(hits[0].drift < 1e-10);

        cleanup(&dir);
    }

    #[test]
    fn genome_builds() {
        let dir = test_dir("genome");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();
        t.build_genome().unwrap();
        assert!(t.genome_depth().unwrap() >= 1);
        assert!(t.genome_codons().unwrap() >= 1);

        cleanup(&dir);
    }

    #[test]
    fn scale_100k() {
        let dir = test_dir("scale");
        let schema = vec![
            ColumnDef { name: "id".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
            ColumnDef { name: "name".into(), col_type: ColumnType::Bytes, indexed: false, not_null: false, unique: false },
            ColumnDef { name: "age".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
            ColumnDef { name: "city".into(), col_type: ColumnType::Bytes, indexed: true, not_null: false, unique: false },
            ColumnDef { name: "score".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
        ];

        let cities = [b"Tokyo".to_vec(), b"Paris".to_vec(), b"Cairo".to_vec(), b"Lima".to_vec(), b"Oslo".to_vec()];
        let rows: Vec<Vec<ColumnValue>> = (0..100_000)
            .map(|i| {
                vec![
                    ColumnValue::F64(i as f64),
                    ColumnValue::Bytes(format!("user_{:08}", i).into_bytes()),
                    ColumnValue::F64(20.0 + (i % 60) as f64),
                    ColumnValue::Bytes(cities[i % 5].clone()),
                    ColumnValue::F64((i * 17 % 1000) as f64 / 10.0),
                ]
            })
            .collect();

        let mut t = Table::create(&dir, schema).unwrap();
        t.insert_many(&rows).unwrap();
        assert_eq!(t.count(), 100_000);

        // Filter by city
        let tokyo = t.filter_equals("city", b"Tokyo").unwrap();
        assert_eq!(tokyo.len(), 20_000);

        let hits = t
            .search(
                &[
                    ColumnValue::F64(1234.0),
                    ColumnValue::Bytes(b"user_00001234".to_vec()),
                    ColumnValue::F64(54.0),
                    ColumnValue::Bytes(b"Oslo".to_vec()),
                    ColumnValue::F64(97.8),
                ],
                1,
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].drift.is_finite());

        drop(t);
        cleanup(&dir);
    }

    #[test]
    fn update_row() {
        let dir = test_dir("update");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();

        t.update(
            1,
            &[
                ColumnValue::Bytes(b"Beatrice".to_vec()),
                ColumnValue::F64(26.0),
                ColumnValue::Bytes(b"Lima".to_vec()),
                ColumnValue::F64(95.0),
            ],
        )
        .unwrap();

        assert_eq!(t.get_field_bytes(1, 0).unwrap(), b"Beatrice");
        assert_eq!(t.get_field_f64(1, 1).unwrap(), 26.0);
        assert_eq!(t.get_field_bytes(1, 2).unwrap(), b"Lima");
        assert_eq!(t.get_field_f64(1, 3).unwrap(), 95.0);

        cleanup(&dir);
    }

    #[test]
    fn delete_row() {
        let dir = test_dir("delete");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();

        let id_before = t.identity();
        t.delete(1).unwrap();

        // Count stays the same — tombstone delete.
        // The row exists but its quaternion is identity (contributes nothing).
        assert_eq!(t.count(), 5);

        // Table identity changed because the composition changed.
        let id_after = t.identity();
        let diff: f64 = id_before.iter().zip(&id_after).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "identity should change after delete");

        // The deleted row's quaternion is identity — it no longer
        // contributes to the table's composition.
        let deleted_q = t.tree.get(1);
        assert!((deleted_q[0] - 1.0).abs() < 1e-10);
        assert!(deleted_q[1].abs() < 1e-10);
        assert!(deleted_q[2].abs() < 1e-10);
        assert!(deleted_q[3].abs() < 1e-10);

        cleanup(&dir);
    }

    #[test]
    fn not_null_rejects_empty_bytes() {
        let dir = test_dir("not_null_bytes");
        let schema = vec![
            ColumnDef { name: "name".into(), col_type: ColumnType::Bytes, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "age".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
        ];
        let mut t = Table::create(&dir, schema).unwrap();
        // Valid insert works
        assert!(t.insert(&[ColumnValue::Bytes(b"Alice".to_vec()), ColumnValue::F64(30.0)]).is_ok());
        // Empty bytes rejected
        assert!(t.insert(&[ColumnValue::Bytes(b"".to_vec()), ColumnValue::F64(25.0)]).is_err());
        assert_eq!(t.count(), 1);
        cleanup(&dir);
    }

    #[test]
    fn not_null_rejects_nan() {
        let dir = test_dir("not_null_nan");
        let schema = vec![
            ColumnDef { name: "score".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
        ];
        let mut t = Table::create(&dir, schema).unwrap();
        assert!(t.insert(&[ColumnValue::F64(99.0)]).is_ok());
        assert!(t.insert(&[ColumnValue::F64(f64::NAN)]).is_err());
        assert_eq!(t.count(), 1);
        cleanup(&dir);
    }

    #[test]
    fn unique_rejects_duplicate_bytes() {
        let dir = test_dir("unique_bytes");
        let schema = vec![
            ColumnDef { name: "name".into(), col_type: ColumnType::Bytes, indexed: true, not_null: false, unique: true },
            ColumnDef { name: "age".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: false },
        ];
        let mut t = Table::create(&dir, schema).unwrap();
        assert!(t.insert(&[ColumnValue::Bytes(b"Alice".to_vec()), ColumnValue::F64(30.0)]).is_ok());
        assert!(t.insert(&[ColumnValue::Bytes(b"Bob".to_vec()), ColumnValue::F64(25.0)]).is_ok());
        // Duplicate name rejected
        assert!(t.insert(&[ColumnValue::Bytes(b"Alice".to_vec()), ColumnValue::F64(35.0)]).is_err());
        assert_eq!(t.count(), 2);
        cleanup(&dir);
    }

    #[test]
    fn unique_rejects_duplicate_f64() {
        let dir = test_dir("unique_f64");
        let schema = vec![
            ColumnDef { name: "id".into(), col_type: ColumnType::F64, indexed: false, not_null: false, unique: true },
        ];
        let mut t = Table::create(&dir, schema).unwrap();
        assert!(t.insert(&[ColumnValue::F64(1.0)]).is_ok());
        assert!(t.insert(&[ColumnValue::F64(2.0)]).is_ok());
        // Duplicate id rejected
        assert!(t.insert(&[ColumnValue::F64(1.0)]).is_err());
        assert_eq!(t.count(), 2);
        cleanup(&dir);
    }

    #[test]
    fn audit_clean_table() {
        let dir = test_dir("audit_clean");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();

        let result = t.audit().unwrap();
        assert!(result.ok, "clean table should pass audit");
        assert!(result.drift < 1e-10);
        assert!(result.bad_row.is_none());

        cleanup(&dir);
    }

    #[test]
    fn audit_detects_corruption() {
        let dir = test_dir("audit_corrupt");
        let mut t = Table::create(&dir, people_schema()).unwrap();
        t.insert_many(&people_rows()).unwrap();
        t.save().unwrap();

        // Corrupt: manually overwrite a value in col_age.bin
        // Row 2 (Charlie, age 35.0) — change age to 99.0 on disk
        // but don't update the running product
        {
            let age_path = dir.join("col_age.bin");
            let mut f = OpenOptions::new().write(true).open(&age_path).unwrap();
            f.seek(SeekFrom::Start(2 * 8)).unwrap(); // row 2
            f.write_all(&99.0f64.to_le_bytes()).unwrap();
            f.sync_all().unwrap();
        }

        let result = t.audit().unwrap();
        assert!(!result.ok, "corrupted table should fail audit");
        assert!(result.drift > 0.01);
        // Should localize to row 2 (the corrupted row)
        assert_eq!(result.bad_row, Some(2), "should find corruption at row 2");

        cleanup(&dir);
    }
}
