//! Error types for the kdmd-snn library.

use thiserror::Error;

/// Errors produced by kdmd-snn operations.
#[derive(Debug, Error)]
pub enum SnnError {
    /// Matrix, vector, or state dimensions are inconsistent with the operation.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// A configuration or constructor argument is invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A numerical gate failed (non-real operator, unstable fit, structure
    /// mismatch).
    #[error("numerical error: {0}")]
    NumericalError(String),

    /// Dataset loading / IO failure (the `datasets` feature).
    #[error("dataset error: {0}")]
    Dataset(String),

    /// An error propagated from the koopman-dmd identification layer.
    #[error(transparent)]
    Dmd(#[from] koopman_dmd::DmdError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dmd_error_converts_via_from() {
        let dmd_err = koopman_dmd::DmdError::InvalidInput("test".into());
        let snn_err: SnnError = dmd_err.into();
        assert!(matches!(snn_err, SnnError::Dmd(_)));
    }

    #[test]
    fn errors_display_their_context() {
        let err = SnnError::DimensionMismatch("expected 4 rows, got 2".into());
        assert_eq!(
            err.to_string(),
            "dimension mismatch: expected 4 rows, got 2"
        );
    }
}
