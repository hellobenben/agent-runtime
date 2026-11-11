use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Stop,
    Length,
    MaxSteps,
    Credential,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    Model(String),
    Disconnected,
    Persist(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(message) | Self::Persist(message) => write!(f, "{message}"),
            Self::Disconnected => write!(f, "client disconnected"),
        }
    }
}

impl std::error::Error for RunError {}
