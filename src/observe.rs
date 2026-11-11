use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_llm::types::{GenerateFuture, LLMProvider, ModelTurn, TextSink};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Ok,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpan {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub latency: Duration,
    pub finish_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpan {
    pub call_id: String,
    pub name: String,
    pub duration: Duration,
    pub status: ToolStatus,
}

pub trait RunObserver: Send + Sync {
    fn on_model(&self, span: ModelSpan);
    fn on_tool(&self, span: ToolSpan);
}

pub struct NoopObserver;

impl RunObserver for NoopObserver {
    fn on_model(&self, _span: ModelSpan) {}
    fn on_tool(&self, _span: ToolSpan) {}
}

pub struct MeteredModel<'a> {
    pub inner: &'a dyn LLMProvider,
    pub observer: Arc<dyn RunObserver>,
}

impl LLMProvider for MeteredModel<'_> {
    fn generate(
        &self,
        messages: Vec<agent_llm::message::Message>,
        tools: Vec<agent_llm::message::ToolDefinition>,
        sink: Box<dyn TextSink + Send>,
    ) -> GenerateFuture<'_> {
        let started = Instant::now();
        let pending = self.inner.generate(messages, tools, sink);
        Box::pin(async move {
            let turn = pending.await?;
            self.observer.on_model(model_span(&turn, started.elapsed()));
            Ok(turn)
        })
    }

    fn complete(
        &self,
        messages: Vec<agent_llm::message::Message>,
        tools: Vec<agent_llm::message::ToolDefinition>,
    ) -> GenerateFuture<'_> {
        let started = Instant::now();
        let pending = self.inner.complete(messages, tools);
        Box::pin(async move {
            let turn = pending.await?;
            self.observer.on_model(model_span(&turn, started.elapsed()));
            Ok(turn)
        })
    }
}

fn model_span(turn: &ModelTurn, latency: Duration) -> ModelSpan {
    let (input_tokens, output_tokens) = match &turn.usage {
        Some(usage) => (Some(usage.input_tokens), Some(usage.output_tokens)),
        None => (None, None),
    };
    ModelSpan {
        input_tokens,
        output_tokens,
        latency,
        finish_reason: turn.finish_reason.as_str().to_string(),
    }
}
