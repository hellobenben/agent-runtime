use agent_llm::types::TextSink;

/// 旁路事件出口：把 agent 循环里发生的事件转发给外部观察者
/// （流式前端、日志、metrics 等）。所有方法都是 fire-and-forget，
/// 不返回错误：发送失败由实现层自行处理，runtime 不感知。
pub trait Reporter: Send + Sync {
    fn text_sink(&self) -> Box<dyn TextSink + Send>;

    fn on_tool_call(&self, id: &str, name: &str, arguments: &str);

    fn on_tool_result(&self, id: &str, name: &str, result: &str, is_error: bool);
}
