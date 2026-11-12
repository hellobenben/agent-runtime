# agent-runtime

ReAct loop over an [`agent-llm`](https://crates.io/crates/agent-llm) provider. Build a runtime with its loop limit, tool runner, and transcript writer. Each call supplies the conversation, tool definitions, and the reporter. The scope type belongs to the runner and the writer.

```rust
let runtime = agent_runtime::ReactRuntime::new(agent_runtime::DEFAULT_MAX_TURNS, runner, writer);
let mut context = agent_runtime::ConversationContext::new();
context.append_user("hi");
let reason = runtime.run(agent_runtime::RunInput {
    model: &model,
    context: &mut context,
    tools,
    reporter: &reporter,
    scope: &scope,
}).await?;
```

`StopReason` is `Stop`, `Length`, `MaxSteps`, or `Credential`. The loop does not load or save anything itself.
