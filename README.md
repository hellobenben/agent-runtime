# agent-runtime

ReAct loop over an [`agent-llm`](https://crates.io/crates/agent-llm) provider. The caller supplies the messages, tool definitions, and the ports that execute tools, record the transcript, and report events.

```rust
let reason = agent_runtime::run(
    &model,
    &runner,
    messages,
    tools,
    &reporter,
    &writer,
    agent_runtime::DEFAULT_MAX_TURNS,
).await?;
```

`StopReason` is `Stop`, `Length`, `MaxSteps`, or `Credential`. The loop does not load or save anything itself.
