//! Error type.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors returned by xarray-rs.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A dimension name was not found.
    #[error("dimension {0:?} not found (available: {1:?})")]
    DimNotFound(String, Vec<String>),
    /// A coordinate or variable name was not found.
    #[error("no variable or coordinate named {0:?}")]
    NotFound(String),
    /// A label was not found in an index.
    #[error("label {label} not found in index {dim:?}")]
    LabelNotFound {
        /// Dimension being indexed.
        dim: String,
        /// Label rendered as text.
        label: String,
    },
    /// An integer position was out of bounds.
    #[error("index {index} is out of bounds for dimension {dim:?} with size {size}")]
    OutOfBounds {
        /// Dimension being indexed.
        dim: String,
        /// The requested position.
        index: isize,
        /// The dimension length.
        size: usize,
    },
    /// Sizes or shapes are inconsistent.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Two objects could not be aligned.
    #[error("alignment error: {0}")]
    Align(String),
    /// Wrong argument or unsupported combination.
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// A data type does not support the requested operation.
    #[error("dtype error: {0}")]
    Dtype(String),
    /// Merge conflict between variables.
    #[error("merge conflict: {0}")]
    Merge(String),
    /// I/O or format error.
    #[error("io error: {0}")]
    Io(String),
}

impl Error {
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }
    pub(crate) fn shape(msg: impl Into<String>) -> Self {
        Error::Shape(msg.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}
