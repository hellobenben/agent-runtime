use std::future::Future;
use std::pin::Pin;

use agent_llm::message::Message;
use agent_llm::types::Usage;

pub type WriteFuture<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;

pub trait TranscriptWriter: Send + Sync {
    fn on_assistant<'a>(
        &'a self,
        message: &'a Message,
        finish_reason: &'a str,
        usage: Option<&'a Usage>,
    ) -> WriteFuture<'a>;

    fn on_tool<'a>(&'a self, message: &'a Message, is_error: bool) -> WriteFuture<'a>;
}

pub struct NoopWriter;

impl TranscriptWriter for NoopWriter {
    fn on_assistant<'a>(
        &'a self,
        _message: &'a Message,
        _finish_reason: &'a str,
        _usage: Option<&'a Usage>,
    ) -> WriteFuture<'a> {
        Box::pin(async { Ok(()) })
    }

    fn on_tool<'a>(&'a self, _message: &'a Message, _is_error: bool) -> WriteFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}
