use std::future::Future;
use std::pin::Pin;

use agent_llm::message::ToolCall;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResponse {
    pub call_id: String,
    pub name: String,
    pub output: String,
    pub is_error: bool,
    pub credential_failure: bool,
}

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = ToolResponse> + Send + 'a>>;

pub trait ToolRunner: Send + Sync {
    fn execute<'a>(&'a self, call: &'a ToolCall) -> ToolFuture<'a>;
}
