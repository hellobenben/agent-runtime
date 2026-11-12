use std::future::Future;
use std::pin::Pin;

use agent_llm::message::{Message, Role, ToolCall, ToolDefinition};
use agent_llm::types::{LLMProvider, ModelFinish};

use crate::context::ConversationContext;
use crate::error::{RunError, StopReason};
use crate::reporter::Reporter;
use crate::tool::{ToolResponse, ToolRunner};
use crate::transcript::TranscriptWriter;

pub const DEFAULT_MAX_TURNS: u32 = 8;

pub type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<StopReason, RunError>> + Send + 'a>>;

pub struct RunInput<'a, S> {
    pub model: &'a dyn LLMProvider,
    pub context: &'a mut ConversationContext,
    pub tools: Vec<ToolDefinition>,
    pub reporter: &'a dyn Reporter,
    pub scope: &'a S,
}

pub trait Runtime: Send + Sync {
    type Scope: Send + Sync;

    fn run<'a>(&'a self, input: RunInput<'a, Self::Scope>) -> RunFuture<'a>;
}

pub struct ReactRuntime<R, W> {
    max_turns: u32,
    runner: R,
    writer: W,
}

impl<R, W> ReactRuntime<R, W> {
    pub fn new(max_turns: u32, runner: R, writer: W) -> Self {
        Self { max_turns, runner, writer }
    }
}

impl<R, W> Runtime for ReactRuntime<R, W>
where
    R: ToolRunner,
    W: TranscriptWriter<Scope = R::Scope>,
{
    type Scope = R::Scope;

    fn run<'a>(&'a self, input: RunInput<'a, Self::Scope>) -> RunFuture<'a> {
        Box::pin(react(self, input))
    }
}

async fn react<R, W>(runtime: &ReactRuntime<R, W>, input: RunInput<'_, R::Scope>) -> Result<StopReason, RunError>
where
    R: ToolRunner,
    W: TranscriptWriter<Scope = R::Scope>,
{
    let RunInput { model, context, tools, reporter, scope } = input;
    let mut completed = 0u32;
    loop {
        if completed == runtime.max_turns {
            return Ok(StopReason::MaxSteps);
        }
        let turn = model
            .generate(context.messages(), tools.clone(), reporter.text_sink())
            .await
            .map_err(|err| RunError::Model(err.to_string()))?;
        completed += 1;
        let finish = turn.finish_reason;
        let calls = turn.message.tool_calls.clone();
        runtime
            .writer
            .on_assistant(scope, &turn.message, finish.as_str(), turn.usage.as_ref())
            .await
            .map_err(RunError::Persist)?;
        context.append_assistant(&turn.message.content, turn.message.tool_calls);
        if finish == ModelFinish::Length {
            return Ok(StopReason::Length);
        }
        if finish != ModelFinish::ToolCalls || calls.is_empty() {
            return Ok(StopReason::Stop);
        }
        let mut credential_failed = false;
        for call in calls {
            let response = if !call.arguments.is_object() {
                rejected(&call, "arguments must be an object", false)
            } else if credential_failed {
                rejected(&call, "credential rejected", true)
            } else {
                let arguments = serde_json::to_string(&call.arguments).unwrap_or_else(|_| "null".to_string());
                reporter.on_tool_call(&call.id, &call.name, &arguments);
                let response = runtime.runner.execute(scope, &call).await;
                if response.credential_failure {
                    credential_failed = true;
                }
                reporter.on_tool_result(
                    &response.call_id,
                    &response.name,
                    &response.output,
                    response.is_error,
                );
                response
            };
            let tool_message = Message {
                role: Role::Tool,
                content: response.output.clone(),
                tool_calls: Vec::new(),
                tool_call_id: response.call_id.clone(),
            };
            runtime.writer.on_tool(scope, &tool_message, response.is_error).await.map_err(RunError::Persist)?;
            context.append_tool(&response.call_id, &response.output);
        }
        if credential_failed {
            return Ok(StopReason::Credential);
        }
    }
}

fn rejected(call: &ToolCall, output: &str, credential_failure: bool) -> ToolResponse {
    ToolResponse {
        call_id: call.id.clone(),
        name: call.name.clone(),
        output: output.to_string(),
        is_error: true,
        credential_failure,
    }
}
