//! Balanced composition tree on S³ — persisted as a geometric pyramid.
//!
//! Leaves are row quaternions. Each parent is the composition of its
//! children (Hamilton product, left to right). Root is the table identity.
//!
//! On disk: one flat file (tree.q) containing the entire node array.
//! Root at offset 0, levels descend contiguously. Each node is 32 bytes.
//!
//! - open: read root (32 bytes), identity ready
//! - check(): root only, O(1)
//! - update/delete: read+write one branch, O(log n)
//! - full load: mmap or read entire file, O(n)
//!
//! This is a segment tree where the combine operation is quaternion
//! composition. Non-commutative: left child composes before right child.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::groups::sphere::SphereGroup;
use crate::groups::LieGroup;

const DIM: usize = 4;
const NODE_BYTES: usize = DIM * 8; // 32 bytes per quaternion
const IDENTITY: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

/// A balanced binary tree of quaternion compositions, backed by disk.
pub struct CompositionTree {
    /// Flat array: node 0 = root, children of i are 2i+1 and 2i+2.
    nodes: Vec<[f64; 4]>,
    /// Number of actual leaves (rows).
    n: usize,
    /// Number of leaf slots (next power of 2 >= n).
    capacity: usize,
    /// Disk file backing the tree. None for in-memory only.
    file: Option<File>,
    group: SphereGroup,
}

impl CompositionTree {
    /// Build from leaf elements. Computes all internal nodes bottom-up.
    pub fn from_elements(elements: &[[f64; 4]]) -> Self {
        let n = elements.len();
        let capacity = if n == 0 { 1 } else { n.next_power_of_two() };
        let size = 2 * capacity - 1;
        let mut nodes = vec![IDENTITY; size];
        let group = SphereGroup;

        let leaf_start = capacity - 1;
        for (i, q) in elements.iter().enumerate() {
            nodes[leaf_start + i] = *q;
        }

        if size > 1 {
            let mut buf = [0.0; DIM];
            for i in (0..leaf_start).rev() {
                group.compose_into(&nodes[2 * i + 1], &nodes[2 * i + 2], &mut buf);
                nodes[i] = buf;
            }
        }

        Self { nodes, n, capacity, file: None, group }
    }

    /// Empty tree.
    pub fn new() -> Self {
        Self {
            nodes: vec![IDENTITY],
            n: 0,
            capacity: 1,
            file: None,
            group: SphereGroup,
        }
    }

    /// Persist the entire tree to disk. Creates or overwrites tree.q.
    pub fn save_to(&self, path: &Path) -> io::Result<()> {
        let mut f = File::create(path)?;
        for node in &self.nodes {
            for v in node {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Load from disk. Reads the entire tree file.
    pub fn load_from(path: &Path, n: usize) -> io::Result<Self> {
        let capacity = if n == 0 { 1 } else { n.next_power_of_two() };
        let size = 2 * capacity - 1;
        let mut f = OpenOptions::new().read(true).write(true).open(path)?;
        let file_len = f.metadata()?.len() as usize;
        let expected = size * NODE_BYTES;

        if file_len < expected {
            // Tree file is short — rebuild needed
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tree.q too small: {} < {}", file_len, expected),
            ));
        }

        let mut buf = vec![0u8; expected];
        f.seek(SeekFrom::Start(0))?;
        f.read_exact(&mut buf)?;

        let mut nodes = Vec::with_capacity(size);
        for i in 0..size {
            let off = i * NODE_BYTES;
            nodes.push([
                f64::from_le_bytes(buf[off..off + 8].try_into().unwrap()),
                f64::from_le_bytes(buf[off + 8..off + 16].try_into().unwrap()),
                f64::from_le_bytes(buf[off + 16..off + 24].try_into().unwrap()),
                f64::from_le_bytes(buf[off + 24..off + 32].try_into().unwrap()),
            ]);
        }

        Ok(Self {
            nodes,
            n,
            capacity,
            file: Some(f),
            group: SphereGroup,
        })
    }

    /// Load ONLY the root from disk. O(1). For check() and identity().
    pub fn load_root_only(path: &Path) -> io::Result<[f64; 4]> {
        let mut f = File::open(path)?;
        let mut buf = [0u8; NODE_BYTES];
        f.read_exact(&mut buf)?;
        Ok([
            f64::from_le_bytes(buf[0..8].try_into().unwrap()),
            f64::from_le_bytes(buf[8..16].try_into().unwrap()),
            f64::from_le_bytes(buf[16..24].try_into().unwrap()),
            f64::from_le_bytes(buf[24..32].try_into().unwrap()),
        ])
    }

    /// The root — composition of all leaves. O(1).
    pub fn root(&self) -> [f64; 4] {
        self.nodes[0]
    }

    /// Get leaf value at position.
    pub fn get(&self, pos: usize) -> [f64; 4] {
        self.nodes[self.capacity - 1 + pos]
    }

    /// Update one leaf and recompose up to root. O(log n).
    /// Also writes the changed nodes to disk if file is open.
    pub fn update(&mut self, pos: usize, value: [f64; 4]) -> io::Result<()> {
        let mut idx = self.capacity - 1 + pos;
        self.nodes[idx] = value;
        self.write_node(idx)?;

        let mut buf = [0.0; DIM];
        while idx > 0 {
            let parent = (idx - 1) / 2;
            self.group.compose_into(
                &self.nodes[2 * parent + 1],
                &self.nodes[2 * parent + 2],
                &mut buf,
            );
            self.nodes[parent] = buf;
            self.write_node(parent)?;
            idx = parent;
        }
        Ok(())
    }

    /// Append one leaf. May rebuild if capacity exceeded.
    pub fn append(&mut self, value: [f64; 4]) {
        if self.n >= self.capacity {
            let mut elements: Vec<[f64; 4]> = (0..self.n)
                .map(|i| self.nodes[self.capacity - 1 + i])
                .collect();
            elements.push(value);
            let file = self.file.take();
            *self = Self::from_elements(&elements);
            self.file = file;
        } else {
            let idx = self.capacity - 1 + self.n;
            self.nodes[idx] = value;
            self.n += 1;
            let mut i = idx;
            let mut buf = [0.0; DIM];
            while i > 0 {
                let parent = (i - 1) / 2;
                self.group.compose_into(
                    &self.nodes[2 * parent + 1],
                    &self.nodes[2 * parent + 2],
                    &mut buf,
                );
                self.nodes[parent] = buf;
                i = parent;
            }
        }
    }

    /// Number of leaves.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Borrow the active leaf slice without cloning.
    pub fn leaves_slice(&self) -> &[[f64; 4]] {
        let leaf_start = self.capacity - 1;
        &self.nodes[leaf_start..leaf_start + self.n]
    }

    /// All leaf values.
    pub fn leaves(&self) -> Vec<[f64; 4]> {
        self.leaves_slice().to_vec()
    }

    /// Composition of leaves in range [lo, hi). O(log n).
    pub fn prefix_product(&self, t: usize) -> [f64; 4] {
        if t == 0 {
            return IDENTITY;
        }
        if t >= self.n {
            return self.root();
        }
        self.range_product(0, t)
    }

    fn range_product(&self, lo: usize, hi: usize) -> [f64; 4] {
        let mut result = IDENTITY;
        let mut buf = [0.0; DIM];
        let mut l = self.capacity - 1 + lo;
        let mut r = self.capacity - 1 + hi;
        let mut left_parts: Vec<[f64; 4]> = Vec::new();
        let mut right_parts: Vec<[f64; 4]> = Vec::new();

        while l < r {
            if l % 2 == 0 {
                left_parts.push(self.nodes[l]);
                l += 1;
            }
            if r % 2 == 0 {
                r -= 1;
                right_parts.push(self.nodes[r]);
            }
            l = (l - 1) / 2;
            r = (r - 1) / 2;
        }

        for q in &left_parts {
            self.group.compose_into(&result, q, &mut buf);
            result = buf;
        }
        for q in right_parts.iter().rev() {
            self.group.compose_into(&result, q, &mut buf);
            result = buf;
        }
        result
    }

    /// Write one node to the backing file (if open).
    fn write_node(&mut self, idx: usize) -> io::Result<()> {
        if let Some(ref mut f) = self.file {
            f.seek(SeekFrom::Start(idx as u64 * NODE_BYTES as u64))?;
            for v in &self.nodes[idx] {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Attach a file for incremental writes (update/delete).
    pub fn attach_file(&mut self, path: &Path) -> io::Result<()> {
        self.file = Some(OpenOptions::new().read(true).write(true).open(path)?);
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        if let Some(ref mut f) = self.file {
            f.sync_all()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_matches_sequential_composition() {
        let group = SphereGroup;
        let elements: Vec<[f64; 4]> = vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.707, 0.0, 0.0, 0.707],
        ];
        let tree = CompositionTree::from_elements(&elements);
        let mut running = IDENTITY;
        let mut buf = [0.0; DIM];
        for q in &elements {
            group.compose_into(&running, q, &mut buf);
            running = buf;
        }
        let root = tree.root();
        for i in 0..4 {
            assert!((root[i] - running[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn update_changes_root() {
        let elements: Vec<[f64; 4]> = vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let mut tree = CompositionTree::from_elements(&elements);
        let old_root = tree.root();
        tree.update(1, IDENTITY).unwrap();
        let new_root = tree.root();
        let diff: f64 = old_root.iter().zip(&new_root).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01);
    }

    #[test]
    fn update_correctness_10k() {
        let group = SphereGroup;
        let n = 10_000;
        let elements: Vec<[f64; 4]> = (0..n)
            .map(|i| {
                let angle = i as f64 * 0.01;
                let norm = (angle.cos() * angle.cos() + angle.sin() * angle.sin()).sqrt();
                [angle.cos() / norm, angle.sin() / norm, 0.0, 0.0]
            })
            .collect();
        let mut tree = CompositionTree::from_elements(&elements);
        tree.update(5000, IDENTITY).unwrap();

        let mut expected = IDENTITY;
        let mut buf = [0.0; DIM];
        for i in 0..n {
            let q = if i == 5000 { IDENTITY } else { elements[i] };
            group.compose_into(&expected, &q, &mut buf);
            expected = buf;
        }
        let root = tree.root();
        for i in 0..4 {
            assert!((root[i] - expected[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn append_works() {
        let mut tree = CompositionTree::new();
        tree.append([0.5, 0.5, 0.5, 0.5]);
        tree.append([0.0, 1.0, 0.0, 0.0]);
        assert_eq!(tree.len(), 2);

        let group = SphereGroup;
        let mut expected = IDENTITY;
        let mut buf = [0.0; DIM];
        group.compose_into(&expected, &[0.5, 0.5, 0.5, 0.5], &mut buf);
        expected = buf;
        group.compose_into(&expected, &[0.0, 1.0, 0.0, 0.0], &mut buf);
        expected = buf;
        let root = tree.root();
        for i in 0..4 {
            assert!((root[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn delete_via_identity() {
        let elements: Vec<[f64; 4]> = vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let mut tree = CompositionTree::from_elements(&elements);
        tree.update(1, IDENTITY).unwrap();

        let group = SphereGroup;
        let mut expected = IDENTITY;
        let mut buf = [0.0; DIM];
        group.compose_into(&expected, &elements[0], &mut buf);
        expected = buf;
        group.compose_into(&expected, &elements[2], &mut buf);
        expected = buf;
        let root = tree.root();
        for i in 0..4 {
            assert!((root[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn persist_and_reload() {
        let elements: Vec<[f64; 4]> = vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let tree = CompositionTree::from_elements(&elements);
        let root_before = tree.root();

        let dir = std::env::temp_dir().join(format!("ctree_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tree.q");

        tree.save_to(&path).unwrap();
        let loaded = CompositionTree::load_from(&path, 3).unwrap();
        let root_after = loaded.root();

        for i in 0..4 {
            assert!((root_before[i] - root_after[i]).abs() < 1e-15);
        }

        // Leaves match too
        for i in 0..3 {
            let orig = tree.get(i);
            let reloaded = loaded.get(i);
            for j in 0..4 {
                assert!((orig[j] - reloaded[j]).abs() < 1e-15);
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn root_only_load() {
        let elements: Vec<[f64; 4]> = vec![
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
        ];
        let tree = CompositionTree::from_elements(&elements);
        let root = tree.root();

        let dir = std::env::temp_dir().join(format!("ctree_root_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tree.q");
        tree.save_to(&path).unwrap();

        let root_loaded = CompositionTree::load_root_only(&path).unwrap();
        for i in 0..4 {
            assert!((root[i] - root_loaded[i]).abs() < 1e-15);
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
