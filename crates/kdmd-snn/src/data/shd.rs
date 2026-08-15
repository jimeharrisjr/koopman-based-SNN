//! Spiking Heidelberg Digits (SHD) loader — `datasets` feature.
//!
//! SHD (Cramer, Stradmann, Schemmel & Zenke 2020, IEEE TNNLS): spoken digits
//! 0–9 in English and German (20 classes), cochlea-encoded into spike trains
//! over 700 input channels; 8,156 train / 2,264 test samples of ≈ 0.7–1.4 s.
//! Files are HDF5 with variable-length event arrays:
//! `spikes/times` (f64 seconds), `spikes/units` (channel ids), `labels`.
//!
//! Download (see `examples/shd_demo.rs`, which automates this):
//! `https://zenkelab.org/datasets/shd_train.h5.gz` and `shd_test.h5.gz`.

use std::path::Path;

use hdf5_metno as hdf5;

use crate::error::SnnError;

/// One SHD sample: event times (seconds), channel ids (0..700), class label
/// (0..20).
#[derive(Debug, Clone)]
pub struct ShdSample {
    pub times_s: Vec<f64>,
    pub units: Vec<u32>,
    pub label: usize,
}

fn h5err(context: &str, e: impl std::fmt::Display) -> SnnError {
    SnnError::Dataset(format!("{context}: {e}"))
}

/// Load every sample from an SHD HDF5 file.
pub fn load_shd(path: &Path) -> Result<Vec<ShdSample>, SnnError> {
    let file = hdf5::File::open(path).map_err(|e| h5err("opening SHD file", e))?;
    let times_ds = file
        .dataset("spikes/times")
        .map_err(|e| h5err("spikes/times", e))?;
    let units_ds = file
        .dataset("spikes/units")
        .map_err(|e| h5err("spikes/units", e))?;
    let labels_ds = file.dataset("labels").map_err(|e| h5err("labels", e))?;

    let times: Vec<hdf5::types::VarLenArray<f64>> = times_ds
        .read_raw()
        .map_err(|e| h5err("reading spikes/times", e))?;
    // Channel ids ship as varying integer widths across dataset versions.
    let units: Vec<Vec<u32>> = read_vlen_units(&units_ds)?;
    let labels: Vec<i64> = read_labels(&labels_ds)?;

    if times.len() != units.len() || times.len() != labels.len() {
        return Err(SnnError::Dataset(format!(
            "inconsistent SHD file: {} times, {} units, {} labels",
            times.len(),
            units.len(),
            labels.len()
        )));
    }
    let mut out = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        let label = labels[i];
        if !(0..20).contains(&label) {
            return Err(SnnError::Dataset(format!("label {label} out of range")));
        }
        out.push(ShdSample {
            times_s: times[i].iter().copied().collect(),
            units: units[i].clone(),
            label: label as usize,
        });
    }
    Ok(out)
}

fn read_vlen_units(ds: &hdf5::Dataset) -> Result<Vec<Vec<u32>>, SnnError> {
    if let Ok(v) = ds.read_raw::<hdf5::types::VarLenArray<u32>>() {
        return Ok(v.into_iter().map(|a| a.iter().copied().collect()).collect());
    }
    if let Ok(v) = ds.read_raw::<hdf5::types::VarLenArray<u16>>() {
        return Ok(v
            .into_iter()
            .map(|a| a.iter().map(|&x| x as u32).collect())
            .collect());
    }
    if let Ok(v) = ds.read_raw::<hdf5::types::VarLenArray<i64>>() {
        return Ok(v
            .into_iter()
            .map(|a| a.iter().map(|&x| x as u32).collect())
            .collect());
    }
    Err(SnnError::Dataset(
        "spikes/units has an unsupported element type".into(),
    ))
}

fn read_labels(ds: &hdf5::Dataset) -> Result<Vec<i64>, SnnError> {
    if let Ok(v) = ds.read_raw::<i64>() {
        return Ok(v);
    }
    if let Ok(v) = ds.read_raw::<u8>() {
        return Ok(v.into_iter().map(|x| x as i64).collect());
    }
    if let Ok(v) = ds.read_raw::<u32>() {
        return Ok(v.into_iter().map(|x| x as i64).collect());
    }
    Err(SnnError::Dataset("labels has an unsupported type".into()))
}

/// Bin a sample's events into `t_steps` steps of `bin_s` seconds, pooling the
/// 700 channels down to `n_pooled` (contiguous groups). Returns the active
/// pooled channels per step (deduplicated, sorted).
pub fn bin_events(
    sample: &ShdSample,
    n_pooled: usize,
    bin_s: f64,
    t_steps: usize,
) -> Result<Vec<Vec<u32>>, SnnError> {
    if n_pooled == 0 || n_pooled > 700 {
        return Err(SnnError::InvalidParameter(format!(
            "n_pooled must be in 1..=700 (got {n_pooled})"
        )));
    }
    if !crate::util::is_positive(bin_s) || t_steps == 0 {
        return Err(SnnError::InvalidParameter(
            "bin_s and t_steps must be positive".into(),
        ));
    }
    let mut out = vec![Vec::new(); t_steps];
    for (t, &u) in sample.times_s.iter().zip(&sample.units) {
        let step = (t / bin_s) as usize;
        if step >= t_steps {
            continue;
        }
        // Proportional pooling: every pooled channel is reachable even when
        // n_pooled does not divide 700.
        let pooled = ((u as usize * n_pooled) / 700).min(n_pooled - 1) as u32;
        out[step].push(pooled);
    }
    for step in &mut out {
        step.sort_unstable();
        step.dedup();
    }
    Ok(out)
}
