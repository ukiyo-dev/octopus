package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/samber/lo"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

// ConvertToLLMResponse converts an Anthropic Message into the internal unified format.
// Shared by the outbound transformer (non-stream response) and the inbound passthrough aggregator.
func ConvertToLLMResponse(resp *Message) *model.InternalLLMResponse {
	if resp == nil {
		return &model.InternalLLMResponse{Object: "chat.completion"}
	}

	result := &model.InternalLLMResponse{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  resp.Model,
	}

	var (
		content           model.MessageContent
		thinkingText      *string
		thinkingSignature *string
		toolCalls         []model.ToolCall
		textParts         []string
	)

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != nil && *block.Text != "" {
				textParts = append(textParts, *block.Text)
				content.MultipleContent = append(content.MultipleContent, model.MessageContentPart{
					Type: "text",
					Text: block.Text,
				})
			}
		case "tool_use":
			if block.ID != "" && block.Name != nil {
				input := "{}"
				if len(block.Input) > 0 {
					input = string(block.Input)
				}
				toolCalls = append(toolCalls, model.ToolCall{
					ID:   block.ID,
					Type: "function",
					Function: model.FunctionCall{
						Name:      *block.Name,
						Arguments: input,
					},
				})
			}
		case "thinking":
			if block.Thinking != nil {
				thinkingText = block.Thinking
			}
			thinkingSignature = block.Signature
		}
	}

	if len(textParts) > 0 && len(content.MultipleContent) == len(textParts) {
		allText := strings.Join(textParts, "")
		content.Content = &allText
		content.MultipleContent = nil
	}

	result.Choices = []model.Choice{{
		Index: 0,
		Message: &model.Message{
			Role:               resp.Role,
			Content:            content,
			ToolCalls:          toolCalls,
			ReasoningContent:   thinkingText,
			ReasoningSignature: thinkingSignature,
		},
		FinishReason: ConvertStopReason(resp.StopReason),
	}}
	result.Usage = ConvertAnthropicUsage(resp.Usage)

	return result
}

// ConvertStopReason maps Anthropic stop reasons to OpenAI-compatible finish reasons.
func ConvertStopReason(stopReason *string) *string {
	if stopReason == nil {
		return nil
	}
	switch *stopReason {
	case "end_turn":
		return lo.ToPtr("stop")
	case "max_tokens":
		return lo.ToPtr("length")
	case "stop_sequence", "pause_turn":
		return lo.ToPtr("stop")
	case "tool_use":
		return lo.ToPtr("tool_calls")
	case "refusal":
		return lo.ToPtr("content_filter")
	default:
		return stopReason
	}
}

// ConvertAnthropicUsage converts an Anthropic Usage struct to the internal Usage struct.
func ConvertAnthropicUsage(usage *Usage) *model.Usage {
	if usage == nil {
		return nil
	}
	result := &model.Usage{
		PromptTokens:             usage.InputTokens,
		CompletionTokens:         usage.OutputTokens,
		TotalTokens:              usage.InputTokens + usage.OutputTokens + usage.CacheReadInputTokens + usage.CacheCreationInputTokens,
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
		AnthropicUsage:           true,
	}
	if usage.CacheReadInputTokens > 0 {
		result.PromptTokensDetails = &model.PromptTokensDetails{
			CachedTokens: usage.CacheReadInputTokens,
		}
	}
	return result
}

// ConvertFromLLMResponse converts the internal unified response into an Anthropic Message.
func ConvertFromLLMResponse(resp *model.InternalLLMResponse) *Message {
	if resp == nil {
		return &Message{Type: "message", Role: "assistant"}
	}

	msg := &Message{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: resp.Model,
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		var source *model.Message
		if choice.Message != nil {
			source = choice.Message
		} else if choice.Delta != nil {
			source = choice.Delta
		}
		if source != nil {
			if source.Role != "" {
				msg.Role = source.Role
			}
			if source.ReasoningContent != nil && *source.ReasoningContent != "" {
				block := MessageContentBlock{Type: "thinking", Thinking: source.ReasoningContent}
				if source.ReasoningSignature != nil {
					block.Signature = source.ReasoningSignature
				}
				msg.Content = append(msg.Content, block)
			}
			if source.Content.Content != nil && *source.Content.Content != "" {
				msg.Content = append(msg.Content, MessageContentBlock{Type: "text", Text: source.Content.Content})
			} else {
				for _, part := range source.Content.MultipleContent {
					switch part.Type {
					case "text":
						if part.Text != nil {
							msg.Content = append(msg.Content, MessageContentBlock{Type: "text", Text: part.Text})
						}
					}
				}
			}
			for _, tc := range source.ToolCalls {
				input := json.RawMessage("{}")
				if tc.Function.Arguments != "" && json.Valid([]byte(tc.Function.Arguments)) {
					input = json.RawMessage(tc.Function.Arguments)
				}
				name := tc.Function.Name
				msg.Content = append(msg.Content, MessageContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  &name,
					Input: input,
				})
			}
		}
		msg.StopReason = ConvertFinishReason(choice.FinishReason)
	}

	msg.Usage = ConvertUsageToAnthropic(resp.Usage)
	return msg
}

func ConvertFinishReason(finishReason *string) *string {
	if finishReason == nil {
		return nil
	}
	switch *finishReason {
	case "stop":
		return lo.ToPtr("end_turn")
	case "length":
		return lo.ToPtr("max_tokens")
	case "tool_calls":
		return lo.ToPtr("tool_use")
	case "content_filter":
		return lo.ToPtr("refusal")
	default:
		return finishReason
	}
}

func ConvertUsageToAnthropic(usage *model.Usage) *Usage {
	if usage == nil {
		return nil
	}
	result := &Usage{
		InputTokens:             usage.PromptTokens,
		OutputTokens:            usage.CompletionTokens,
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
	}
	if usage.PromptTokensDetails != nil {
		result.CacheReadInputTokens = usage.PromptTokensDetails.CachedTokens
		result.InputTokens -= result.CacheReadInputTokens
	}
	return result
}
