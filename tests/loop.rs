use std::sync::Mutex;

use agent_llm::message::{Message, Role, ToolCall, ToolDefinition};
use agent_llm::types::{GenerateFuture, LLMProvider, ModelFinish, ModelTurn, TextSink, Usage};
use agent_runtime::{Reporter, RunError, StopReason, ToolFuture, ToolResponse, ToolRunner, TranscriptWriter, DEFAULT_MAX_TURNS};

struct Sink;

impl TextSink for Sink {
    fn on_delta(&self, _text: &str) {}
}

struct Script {
    turns: Mutex<Vec<ModelTurn>>,
    seen: Mutex<Vec<Vec<Message>>>,
}

impl LLMProvider for Script {
    fn generate(
        &self,
        messages: Vec<Message>,
        _tools: Vec<ToolDefinition>,
        _sink: Box<dyn TextSink + Send>,
    ) -> GenerateFuture<'_> {
        self.seen.lock().unwrap().push(messages);
        let turn = self.turns.lock().unwrap().remove(0);
        Box::pin(async move { Ok(turn) })
    }

    fn complete(&self, messages: Vec<Message>, tools: Vec<ToolDefinition>) -> GenerateFuture<'_> {
        self.generate(messages, tools, Box::new(Sink))
    }
}

struct FnRunner<F> {
    hits: Mutex<Vec<String>>,
    respond: F,
}

impl<F> ToolRunner for FnRunner<F>
where
    F: Fn(&ToolCall) -> ToolResponse + Send + Sync,
{
    fn execute<'a>(&'a self, call: &'a ToolCall) -> ToolFuture<'a> {
        self.hits.lock().unwrap().push(call.name.clone());
        let response = (self.respond)(call);
        Box::pin(async move { response })
    }
}

struct RecReporter {
    events: Mutex<Vec<String>>,
}

impl Reporter for RecReporter {
    fn text_sink(&self) -> Box<dyn TextSink + Send> {
        Box::new(Sink)
    }

    fn on_tool_call(&self, id: &str, name: &str, arguments: &str) {
        self.events.lock().unwrap().push(format!("call:{id}:{name}:{arguments}"));
    }

    fn on_tool_result(&self, id: &str, name: &str, result: &str, is_error: bool) {
        self.events.lock().unwrap().push(format!("result:{id}:{name}:{result}:{is_error}"));
    }
}

struct RecWriter {
    assistants: Mutex<Vec<(String, String)>>,
    tools: Mutex<Vec<(String, String, bool)>>,
    fail_assistant: Option<String>,
}

impl TranscriptWriter for RecWriter {
    fn on_assistant<'a>(
        &'a self,
        message: &'a Message,
        finish_reason: &'a str,
        _usage: Option<&'a Usage>,
    ) -> agent_runtime::WriteFuture<'a> {
        if let Some(reason) = &self.fail_assistant {
            let reason = reason.clone();
            return Box::pin(async move { Err(reason) });
        }
        self.assistants.lock().unwrap().push((message.content.clone(), finish_reason.to_string()));
        Box::pin(async { Ok(()) })
    }

    fn on_tool<'a>(&'a self, message: &'a Message, is_error: bool) -> agent_runtime::WriteFuture<'a> {
        self.tools.lock().unwrap().push((message.tool_call_id.clone(), message.content.clone(), is_error));
        Box::pin(async { Ok(()) })
    }
}

fn user(content: &str) -> Message {
    Message {
        role: Role::User,
        content: content.to_string(),
        tool_calls: Vec::new(),
        tool_call_id: String::new(),
    }
}

fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: name.to_string(),
        name: name.to_string(),
        arguments,
    }
}

fn turn(content: &str, finish: ModelFinish, calls: Vec<ToolCall>) -> ModelTurn {
    ModelTurn {
        message: Message {
            role: Role::Assistant,
            content: content.to_string(),
            tool_calls: calls,
            tool_call_id: String::new(),
        },
        finish_reason: finish,
        usage: None,
    }
}

fn response(call: &ToolCall, output: &str, is_error: bool, credential_failure: bool) -> ToolResponse {
    ToolResponse {
        call_id: call.id.clone(),
        name: call.name.clone(),
        output: output.to_string(),
        is_error,
        credential_failure,
    }
}

struct Harness {
    script: Script,
    runner: FnRunner<fn(&ToolCall) -> ToolResponse>,
    reporter: RecReporter,
    writer: RecWriter,
}

fn harness(turns: Vec<ModelTurn>, respond: fn(&ToolCall) -> ToolResponse) -> Harness {
    Harness {
        script: Script {
            turns: Mutex::new(turns),
            seen: Mutex::new(Vec::new()),
        },
        runner: FnRunner {
            hits: Mutex::new(Vec::new()),
            respond,
        },
        reporter: RecReporter {
            events: Mutex::new(Vec::new()),
        },
        writer: RecWriter {
            assistants: Mutex::new(Vec::new()),
            tools: Mutex::new(Vec::new()),
            fail_assistant: None,
        },
    }
}

fn ok(call: &ToolCall) -> ToolResponse {
    response(call, "ok", false, false)
}

#[tokio::test]
async fn a_stop_finish_ends_without_calling_tools() {
    let harness = harness(vec![turn("hello", ModelFinish::Stop, vec![])], ok);
    let reason = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![user("hi")],
        vec![],
        &harness.reporter,
        &harness.writer,
        DEFAULT_MAX_TURNS,
    )
    .await
    .unwrap();

    assert_eq!(reason, StopReason::Stop);
    assert!(harness.runner.hits.lock().unwrap().is_empty());
    assert_eq!(harness.writer.assistants.lock().unwrap().as_slice(), [("hello".to_string(), "stop".to_string())]);
}

#[tokio::test]
async fn a_tool_round_returns_to_the_model_and_then_stops() {
    let harness = harness(
        vec![
            turn("", ModelFinish::ToolCalls, vec![call("plain", serde_json::json!({})), call("next", serde_json::json!({}))]),
            turn("done", ModelFinish::Stop, vec![]),
        ],
        |call| {
            if call.name == "plain" {
                response(call, "invalid time range", true, false)
            } else {
                ok(call)
            }
        },
    );
    let reason = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![user("hi")],
        vec![],
        &harness.reporter,
        &harness.writer,
        DEFAULT_MAX_TURNS,
    )
    .await
    .unwrap();

    assert_eq!(reason, StopReason::Stop);
    assert_eq!(harness.runner.hits.lock().unwrap().as_slice(), ["plain", "next"]);
    let seen = harness.script.seen.lock().unwrap();
    assert_eq!(seen[1][0].content, "hi");
    assert_eq!(seen[1][1].role, Role::Assistant);
    assert_eq!(seen[1][2].tool_call_id, "plain");
    assert_eq!(seen[1][2].content, "invalid time range");
    assert_eq!(seen[1][3].tool_call_id, "next");
    assert_eq!(harness.reporter.events.lock().unwrap().as_slice(), [
        "call:plain:plain:{}".to_string(),
        "result:plain:plain:invalid time range:true".to_string(),
        "call:next:next:{}".to_string(),
        "result:next:next:ok:false".to_string(),
    ]);
}

#[tokio::test]
async fn credential_failure_skips_later_tools_and_stops() {
    let harness = harness(
        vec![turn("", ModelFinish::ToolCalls, vec![call("denied", serde_json::json!({})), call("next", serde_json::json!({}))])],
        |call| response(call, "登录已失效", true, true),
    );
    let reason = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![],
        vec![],
        &harness.reporter,
        &harness.writer,
        DEFAULT_MAX_TURNS,
    )
    .await
    .unwrap();

    assert_eq!(reason, StopReason::Credential);
    assert_eq!(harness.runner.hits.lock().unwrap().as_slice(), ["denied"]);
    assert_eq!(harness.script.seen.lock().unwrap().len(), 1);
    assert_eq!(harness.writer.tools.lock().unwrap().as_slice(), [
        ("denied".to_string(), "登录已失效".to_string(), true),
        ("next".to_string(), "credential rejected".to_string(), true),
    ]);
    assert_eq!(harness.reporter.events.lock().unwrap().as_slice(), [
        "call:denied:denied:{}".to_string(),
        "result:denied:denied:登录已失效:true".to_string(),
    ]);
}

#[tokio::test]
async fn the_turn_limit_stops_before_another_model_call() {
    assert_eq!(DEFAULT_MAX_TURNS, 8);
    let harness = harness(
        vec![
            turn("", ModelFinish::ToolCalls, vec![call("once", serde_json::json!({}))]),
            turn("unused", ModelFinish::Stop, vec![]),
        ],
        ok,
    );
    let reason = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![],
        vec![],
        &harness.reporter,
        &harness.writer,
        1,
    )
    .await
    .unwrap();

    assert_eq!(reason, StopReason::MaxSteps);
    assert_eq!(harness.script.seen.lock().unwrap().len(), 1);
    assert_eq!(harness.runner.hits.lock().unwrap().as_slice(), ["once"]);
}

#[tokio::test]
async fn a_length_finish_returns_without_executing_tools() {
    let harness = harness(
        vec![turn("cut", ModelFinish::Length, vec![call("once", serde_json::json!({}))])],
        ok,
    );
    let reason = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![],
        vec![],
        &harness.reporter,
        &harness.writer,
        DEFAULT_MAX_TURNS,
    )
    .await
    .unwrap();

    assert_eq!(reason, StopReason::Length);
    assert!(harness.runner.hits.lock().unwrap().is_empty());
    assert_eq!(harness.writer.assistants.lock().unwrap().as_slice(), [("cut".to_string(), "length".to_string())]);
}

#[tokio::test]
async fn a_writer_failure_is_returned_to_the_caller() {
    let mut harness = harness(vec![turn("hello", ModelFinish::Stop, vec![])], ok);
    harness.writer.fail_assistant = Some("run lock lost".to_string());
    let error = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![],
        vec![],
        &harness.reporter,
        &harness.writer,
        DEFAULT_MAX_TURNS,
    )
    .await
    .unwrap_err();

    assert_eq!(error, RunError::Persist("run lock lost".to_string()));
}

#[tokio::test]
async fn a_non_object_argument_skips_the_runner_and_returns_to_the_model() {
    let harness = harness(
        vec![
            turn("", ModelFinish::ToolCalls, vec![call("bad", serde_json::json!("nope"))]),
            turn("done", ModelFinish::Stop, vec![]),
        ],
        ok,
    );
    let reason = agent_runtime::run(
        &harness.script,
        &harness.runner,
        vec![],
        vec![],
        &harness.reporter,
        &harness.writer,
        DEFAULT_MAX_TURNS,
    )
    .await
    .unwrap();

    assert_eq!(reason, StopReason::Stop);
    assert!(harness.runner.hits.lock().unwrap().is_empty());
    assert!(harness.reporter.events.lock().unwrap().is_empty());
    assert_eq!(
        harness.writer.tools.lock().unwrap().as_slice(),
        [("bad".to_string(), "arguments must be an object".to_string(), true)]
    );
}

#[tokio::test]
async fn a_model_error_is_returned_to_the_caller() {
    struct Down;

    impl LLMProvider for Down {
        fn generate(
            &self,
            _messages: Vec<Message>,
            _tools: Vec<ToolDefinition>,
            _sink: Box<dyn TextSink + Send>,
        ) -> GenerateFuture<'_> {
            Box::pin(async { Err(agent_llm::message::Error::RequestError { msg: "down".to_string() }) })
        }

        fn complete(&self, messages: Vec<Message>, tools: Vec<ToolDefinition>) -> GenerateFuture<'_> {
            self.generate(messages, tools, Box::new(Sink))
        }
    }

    let harness = harness(vec![], ok);
    let error = agent_runtime::run(&Down, &harness.runner, vec![], vec![], &harness.reporter, &harness.writer, DEFAULT_MAX_TURNS)
        .await
        .unwrap_err();

    assert_eq!(error, RunError::Model("request error: down".to_string()));
}
