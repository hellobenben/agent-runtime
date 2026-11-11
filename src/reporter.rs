use std::future::Future;
use std::pin::Pin;

use agent_llm::types::TextSink;

pub type ReportFuture<'a> = Pin<Box<dyn Future<Output = Result<(), ()>> + Send + 'a>>;

pub trait Reporter: Send + Sync {
    fn text_sink(&self) -> Box<dyn TextSink + Send>;

    fn on_tool_call(&self, id: &str, name: &str, arguments: &str) -> ReportFuture<'_>;

    fn on_tool_result(&self, id: &str, name: &str, result: &str, is_error: bool) -> ReportFuture<'_>;

    fn client_gone(&self) -> bool {
        false
    }
}
