use agent_llm::message::{Message, Role, ToolCall};

pub struct ConversationContext {
    messages: Vec<Message>,
}

impl ConversationContext {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    pub fn append_user(&mut self, content: &str) {
        self.messages.push(Message {
            role: Role::User,
            content: content.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: String::new(),
        });
    }

    pub fn append_system(&mut self, content: &str) {
        self.messages.insert(0, Message {
            role: Role::System,
            content: content.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: String::new(),
        })
    }

    pub fn append_assistant(&mut self, content: &str, tool_calls: Vec<ToolCall>) {
        self.messages.push(Message {
            role: Role::Assistant,
            content: content.to_string(),
            tool_calls,
            tool_call_id: String::new(),
        })
    }

    pub fn append_tool(&mut self, call_id: &str, output: &str) {
        self.messages.push(Message {
            role: Role::Tool,
            content: output.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: call_id.to_string(),
        })
    }
}