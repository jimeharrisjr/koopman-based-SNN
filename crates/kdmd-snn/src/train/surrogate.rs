//! Surrogate derivatives for the spike threshold.
//!
//! `Θ'(v − θ)` is zero almost everywhere and undefined at the threshold; the
//! backward pass replaces it with a smooth surrogate evaluated at the
//! pre-reset potential (Neftci, Mostafa & Zenke 2019). The forward pass is
//! untouched.

/// Which surrogate derivative to use.
#[derive(Debug, Clone, Copy)]
pub enum SurrogateKind {
    /// Fast sigmoid / SuperSpike: `σ'(x) = 1 / (1 + β|x|)²`.
    FastSigmoid { beta: f64 },
    /// Arctan-based: `σ'(x) = β / (1 + (π·β·x)²)`.
    Atan { beta: f64 },
    /// Boxcar (straight-through window): `1/(2w)` for `|x| ≤ w`, else 0.
    Boxcar { half_width: f64 },
}

impl SurrogateKind {
    /// The surrogate derivative at `x = v_pre − θ`.
    pub fn derivative(&self, x: f64) -> f64 {
        match *self {
            SurrogateKind::FastSigmoid { beta } => {
                let d = 1.0 + beta * x.abs();
                1.0 / (d * d)
            }
            SurrogateKind::Atan { beta } => {
                let z = std::f64::consts::PI * beta * x;
                beta / (1.0 + z * z)
            }
            SurrogateKind::Boxcar { half_width } => {
                if x.abs() <= half_width {
                    0.5 / half_width
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sigmoid_peaks_at_threshold_and_decays() {
        let s = SurrogateKind::FastSigmoid { beta: 5.0 };
        assert_eq!(s.derivative(0.0), 1.0);
        assert!(s.derivative(0.5) < s.derivative(0.1));
        assert!(s.derivative(-0.5) < s.derivative(-0.1));
        // Symmetric.
        assert_eq!(s.derivative(0.3), s.derivative(-0.3));
    }

    #[test]
    fn atan_and_boxcar_shapes() {
        let a = SurrogateKind::Atan { beta: 2.0 };
        assert!(a.derivative(0.0) > a.derivative(1.0));
        let b = SurrogateKind::Boxcar { half_width: 0.25 };
        assert_eq!(b.derivative(0.2), 2.0);
        assert_eq!(b.derivative(0.3), 0.0);
    }
}
