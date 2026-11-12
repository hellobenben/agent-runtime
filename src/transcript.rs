use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use agent_llm::message::Message;
use agent_llm::types::Usage;

pub type WriteFuture<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;

pub trait TranscriptWriter: Send + Sync {
    type Scope: Send + Sync;

    fn on_assistant<'a>(
        &'a self,
        scope: &'a Self::Scope,
        message: &'a Message,
        finish_reason: &'a str,
        usage: Option<&'a Usage>,
    ) -> WriteFuture<'a>;

    fn on_tool<'a>(&'a self, scope: &'a Self::Scope, message: &'a Message, is_error: bool) -> WriteFuture<'a>;
}

impl<W> TranscriptWriter for &W
where
    W: TranscriptWriter + ?Sized,
{
    type Scope = W::Scope;

    fn on_assistant<'a>(
        &'a self,
        scope: &'a Self::Scope,
        message: &'a Message,
        finish_reason: &'a str,
        usage: Option<&'a Usage>,
    ) -> WriteFuture<'a> {
        (*self).on_assistant(scope, message, finish_reason, usage)
    }

    fn on_tool<'a>(&'a self, scope: &'a Self::Scope, message: &'a Message, is_error: bool) -> WriteFuture<'a> {
        (*self).on_tool(scope, message, is_error)
    }
}

pub struct NoopWriter<S = ()> {
    _scope: PhantomData<S>,
}

impl<S> NoopWriter<S> {
    pub fn new() -> Self {
        Self { _scope: PhantomData }
    }
}

impl<S: Send + Sync> TranscriptWriter for NoopWriter<S> {
    type Scope = S;

    fn on_assistant<'a>(
        &'a self,
        _scope: &'a Self::Scope,
        _message: &'a Message,
        _finish_reason: &'a str,
        _usage: Option<&'a Usage>,
    ) -> WriteFuture<'a> {
        Box::pin(async { Ok(()) })
    }

    fn on_tool<'a>(&'a self, _scope: &'a Self::Scope, _message: &'a Message, _is_error: bool) -> WriteFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}
