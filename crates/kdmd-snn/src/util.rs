//! Small internal helpers.

/// `x` is a finite, strictly positive number. Rejects NaN and ±∞, so
/// parameter validation using this cannot be bypassed by pathological floats.
pub(crate) fn is_positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// `a` and `b` are finite and `a > b`. NaN on either side fails.
pub(crate) fn exceeds(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && a > b
}

/// Every value is finite.
pub(crate) fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_positive_rejects_pathological_floats() {
        assert!(is_positive(0.1));
        assert!(!is_positive(0.0));
        assert!(!is_positive(-1.0));
        assert!(!is_positive(f64::NAN));
        assert!(!is_positive(f64::INFINITY));
    }

    #[test]
    fn exceeds_rejects_nan_and_requires_strict_order() {
        assert!(exceeds(1.0, 0.0));
        assert!(!exceeds(0.0, 0.0));
        assert!(!exceeds(f64::NAN, 0.0));
        assert!(!exceeds(1.0, f64::NAN));
        assert!(!exceeds(f64::INFINITY, 0.0));
    }
}
