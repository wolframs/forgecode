use forge_domain::Transformer;

use crate::dto::openai::{ExtraContent, Message, ReasoningDetail, Request};

/// The literal Google documents for a function call whose thought signature is
/// not available: a Gemini 3 backend skips validation for a call carrying it,
/// at a cost in the model's performance ("strongly discouraged").
/// ref: https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures
pub const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

/// True for a reasoning detail in which OpenRouter-shaped providers carry a
/// Gemini thought signature.
fn is_gemini_signature_detail(detail: &ReasoningDetail) -> bool {
    detail.r#type == "reasoning.encrypted" && detail.format.as_deref() == Some("google-gemini-v1")
}

fn has_gemini_signature_detail(message: &Message) -> bool {
    message
        .reasoning_details
        .as_ref()
        .is_some_and(|details| details.iter().any(is_gemini_signature_detail))
}

fn skip_signature() -> ExtraContent {
    ExtraContent::from(SKIP_THOUGHT_SIGNATURE.to_string())
}

/// Gives the skip value to a tool call that goes out with no thought signature
/// where a Gemini 3 backend requires one: the first call of a step (of
/// parallel calls only the first carries a signature).
///
/// A message whose signatures go back in `reasoning_details` is left alone.
pub struct DefaultThoughtSignature;

impl Transformer for DefaultThoughtSignature {
    type Value = Request;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        for message in request.messages.iter_mut().flatten() {
            if has_gemini_signature_detail(message) {
                continue;
            }
            if let Some(tool_call) = message
                .tool_calls
                .as_mut()
                .and_then(|tool_calls| tool_calls.first_mut())
                && tool_call
                    .extra_content
                    .as_ref()
                    .and_then(ExtraContent::thought_signature)
                    .is_none_or(|signature| signature.is_empty())
            {
                tool_call.extra_content = Some(skip_signature());
            }
        }

        request
    }
}

/// Replaces every thought signature in the request with the skip value, for
/// the one retry after a provider refused the request for its signatures.
///
/// The first call of every step carries the skip value, any other call that
/// had a signature has it replaced, and signatures that are not required
/// (on a message, or in `reasoning_details`) are removed.
pub struct SkipThoughtSignatures;

impl Transformer for SkipThoughtSignatures {
    type Value = Request;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        for message in request.messages.iter_mut().flatten() {
            message.extra_content = None;

            if let Some(details) = message.reasoning_details.as_mut() {
                details.retain(|detail| !is_gemini_signature_detail(detail));
                if details.is_empty() {
                    message.reasoning_details = None;
                }
            }

            for (index, tool_call) in message.tool_calls.iter_mut().flatten().enumerate() {
                if index == 0 || tool_call.extra_content.is_some() {
                    tool_call.extra_content = Some(skip_signature());
                }
            }
        }

        request
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::{Value, json};

    use super::*;

    fn tool_call(id: &str, signature: Option<&str>) -> Value {
        let mut call = json!({
            "id": id,
            "type": "function",
            "function": { "name": "mcp_papertool_tool_search_papers", "arguments": "{}" }
        });
        if let Some(signature) = signature {
            call["extra_content"] = json!({ "google": { "thought_signature": signature } });
        }
        call
    }

    fn assistant(tool_calls: Vec<Value>) -> Value {
        json!({ "role": "assistant", "content": "", "tool_calls": tool_calls })
    }

    fn request(messages: Vec<Value>) -> Request {
        serde_json::from_value(json!({ "model": "gemini-3.1-pro-preview", "messages": messages }))
            .unwrap()
    }

    /// The signatures of each message's tool calls, as they go out.
    fn signatures(request: Request) -> Vec<Vec<Option<String>>> {
        request
            .messages
            .unwrap()
            .into_iter()
            .filter_map(|message| message.tool_calls)
            .map(|tool_calls| {
                tool_calls
                    .iter()
                    .map(|tool_call| {
                        tool_call
                            .extra_content
                            .as_ref()
                            .and_then(ExtraContent::thought_signature)
                    })
                    .collect()
            })
            .collect()
    }

    fn skip() -> Option<String> {
        Some(SKIP_THOUGHT_SIGNATURE.to_string())
    }

    /// Run F, conversation 9b9cdbba: five signed calls, then `call_23281984`
    /// with no signature of its own (messages 10 and 12 as sent).
    #[test]
    fn test_default_gives_unsigned_call_the_skip_value() {
        let fixture = request(vec![
            assistant(vec![tool_call("call_218442", Some("AY89a19J"))]),
            assistant(vec![tool_call("call_23281984", None)]),
        ]);

        let actual = signatures(DefaultThoughtSignature.transform(fixture));

        let expected = vec![vec![Some("AY89a19J".to_string())], vec![skip()]];
        assert_eq!(actual, expected);
    }

    /// The accepted parallel shape: only the first call is signed.
    #[test]
    fn test_default_leaves_later_parallel_calls_unsigned() {
        let fixture = request(vec![
            assistant(vec![
                tool_call("call_1", Some("AY89a18r")),
                tool_call("call_2", None),
            ]),
            assistant(vec![tool_call("call_3", None), tool_call("call_4", None)]),
        ]);

        let actual = signatures(DefaultThoughtSignature.transform(fixture));

        let expected = vec![vec![Some("AY89a18r".to_string()), None], vec![skip(), None]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_default_leaves_message_with_signature_in_reasoning_details() {
        let mut message = assistant(vec![tool_call("call_23281984", None)]);
        message["reasoning_details"] = json!([{
            "type": "reasoning.encrypted",
            "data": "ZTI0ODMwYTctNWNkNi00MmZlLTk5OGItZWU1MzllNzJiOWMz",
            "id": "call_23281984",
            "format": "google-gemini-v1",
            "index": 0
        }]);
        let fixture = request(vec![message]);

        let actual = signatures(DefaultThoughtSignature.transform(fixture));

        let expected = vec![vec![None]];
        assert_eq!(actual, expected);
    }

    /// Request E of the fallback probe, accepted on 2026-10-07: every call's
    /// signature replaced by the skip value.
    #[test]
    fn test_skip_replaces_every_signature() {
        let mut signed_message = assistant(vec![
            tool_call("call_162924", Some("AY89a18r")),
            tool_call("call_2", None),
            tool_call("call_3", Some("AY89a1+F")),
        ]);
        signed_message["extra_content"] = json!({ "google": { "thought_signature": "AY89msg" } });
        let mut detail_message = assistant(vec![tool_call("call_23281984", None)]);
        detail_message["reasoning_details"] = json!([
            { "type": "reasoning.text", "text": "thinking" },
            { "type": "reasoning.encrypted", "data": "ZTI0", "id": "call_23281984",
              "format": "google-gemini-v1", "index": 0 }
        ]);
        let fixture = request(vec![signed_message, detail_message]);

        let actual = SkipThoughtSignatures.transform(fixture);

        let messages = actual.messages.clone().unwrap();
        assert!(messages[0].extra_content.is_none());
        let details = messages[1].reasoning_details.clone().unwrap();
        assert_eq!(details.len(), 1);
        assert_eq!(details[0].r#type, "reasoning.text");
        let expected = vec![vec![skip(), None, skip()], vec![skip()]];
        assert_eq!(signatures(actual), expected);
    }
}
