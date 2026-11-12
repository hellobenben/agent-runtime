use agent_llm::message::{Message, Role, ToolCall, ToolDefinition};
use agent_llm::types::{LLMProvider, ModelFinish};

use crate::error::{RunError, StopReason};
use crate::reporter::Reporter;
use crate::tool::{ToolResponse, ToolRunner};
use crate::transcript::TranscriptWriter;

pub const DEFAULT_MAX_TURNS: u32 = 8;

pub async fn run(
    model: &dyn LLMProvider,
    runner: &dyn ToolRunner,
    mut messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    reporter: &dyn Reporter,
    writer: &dyn TranscriptWriter,
    max_turns: u32,
) -> Result<StopReason, RunError> {
    let mut completed = 0u32;
    loop {
        if completed == max_turns {
            return Ok(StopReason::MaxSteps);
        }
        let turn = model
            .generate(messages.clone(), tools.clone(), reporter.text_sink())
            .await
            .map_err(|err| RunError::Model(err.to_string()))?;
        completed += 1;
        let finish = turn.finish_reason;
        let calls = turn.message.tool_calls.clone();
        writer
            .on_assistant(&turn.message, finish.as_str(), turn.usage.as_ref())
            .await
            .map_err(RunError::Persist)?;
        messages.push(turn.message);
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
                let response = runner.execute(&call).await;
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
            writer.on_tool(&tool_message, response.is_error).await.map_err(RunError::Persist)?;
            messages.push(tool_message);
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
