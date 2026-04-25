package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/bestruirui/octopus/internal/transformer/model"
)
type ChatOutbound struct{}

func (o *ChatOutbound) TargetFormat() model.APIFormat {
	return model.APIFormatOpenAIChatCompletion
}

func (o *ChatOutbound) TransformRequest(ctx context.Context, request *model.InternalLLMRequest, baseUrl, key string) (*http.Request, error) {
	isNativeFormat := request.RawAPIFormat == model.APIFormatOpenAIChatCompletion
	isPassthroughFormat := request.RawAPIFormat == model.APIFormatPassthrough
	passthrough := len(request.RawRequest) > 0 && (isNativeFormat || isPassthroughFormat)

	var body []byte
	var err error

	if passthrough {
		body = patchRawRequest(request.RawRequest, request.Model, isNativeFormat)
	} else {
		request.ClearHelpFields()
		for i := range request.Messages {
			if request.Messages[i].Role == "developer" {
				request.Messages[i].Role = "system"
			}
		}
		if request.Stream != nil && *request.Stream {
			if request.StreamOptions == nil {
				request.StreamOptions = &model.StreamOptions{IncludeUsage: true}
			} else if !request.StreamOptions.IncludeUsage {
				request.StreamOptions.IncludeUsage = true
			}
		}
		body, err = json.Marshal(request)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	parsedUrl, err := url.Parse(strings.TrimSuffix(baseUrl, "/"))
	if err != nil {
		return nil, fmt.Errorf("failed to parse base url: %w", err)
	}
	if passthrough && request.RawPath != "" {
		parsedUrl.Path = parsedUrl.Path + request.RawPath
	} else {
		parsedUrl.Path = parsedUrl.Path + "/chat/completions"
	}
	req.URL = parsedUrl
	req.Method = http.MethodPost
	return req, nil
}

func (o *ChatOutbound) TransformResponse(ctx context.Context, response *http.Response) (*model.InternalLLMResponse, error) {
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("response body is empty")
	}

	var resp model.InternalLLMResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	resp.RawResponse = body
	resp.RawResponseFormat = model.APIFormatOpenAIChatCompletion
	return &resp, nil
}

func (o *ChatOutbound) TransformStream(ctx context.Context, eventData []byte) (*model.InternalLLMResponse, error) {
	if bytes.HasPrefix(eventData, []byte("[DONE]")) {
		return &model.InternalLLMResponse{
			Object:            "[DONE]",
			RawResponse:       eventData,
			RawResponseFormat: model.APIFormatOpenAIChatCompletion,
		}, nil
	}

	var errCheck struct {
		Error *model.ErrorDetail `json:"error"`
	}
	if err := json.Unmarshal(eventData, &errCheck); err == nil && errCheck.Error != nil {
		return nil, &model.ResponseError{
			Detail: *errCheck.Error,
		}
	}

	var resp model.InternalLLMResponse
	if err := json.Unmarshal(eventData, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal stream chunk: %w", err)
	}
	resp.RawResponse = eventData
	resp.RawResponseFormat = model.APIFormatOpenAIChatCompletion
	return &resp, nil
}

func (o *ChatOutbound) ReconstructFromRawSSE(ctx context.Context, rawBytes []byte) (*model.InternalLLMResponse, error) {
	decoder := ssestream.NewDecoder(&http.Response{Body: io.NopCloser(bytes.NewReader(rawBytes))})
	stream := ssestream.NewStream[openai.ChatCompletionChunk](decoder, nil)

	var acc openai.ChatCompletionAccumulator
	for stream.Next() {
		acc.AddChunk(stream.Current())
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}
	if acc.ID == "" && len(acc.Choices) == 0 {
		return nil, nil
	}
	return convertSDKChatCompletion(&acc.ChatCompletion), nil
}

func convertSDKChatCompletion(comp *openai.ChatCompletion) *model.InternalLLMResponse {
	result := &model.InternalLLMResponse{
		ID:      comp.ID,
		Object:  "chat.completion",
		Model:   comp.Model,
		Created: comp.Created,
	}

	choices := make([]model.Choice, 0, len(comp.Choices))
	for _, c := range comp.Choices {
		msg := &model.Message{Role: "assistant"}
		if c.Message.Content != "" {
			msg.Content = model.MessageContent{Content: &c.Message.Content}
		}
		for _, tc := range c.Message.ToolCalls {
			if ftc, ok := tc.AsAny().(openai.ChatCompletionMessageFunctionToolCall); ok {
				msg.ToolCalls = append(msg.ToolCalls, model.ToolCall{
					ID:   ftc.ID,
					Type: "function",
					Function: model.FunctionCall{
						Name:      ftc.Function.Name,
						Arguments: ftc.Function.Arguments,
					},
				})
			}
		}
		var fr *string
		if c.FinishReason != "" {
			s := c.FinishReason
			fr = &s
		}
		choices = append(choices, model.Choice{
			Index:        int(c.Index),
			Message:      msg,
			FinishReason: fr,
		})
	}
	result.Choices = choices

	u := comp.Usage
	if u.TotalTokens > 0 || u.PromptTokens > 0 || u.CompletionTokens > 0 {
		usage := &model.Usage{
			PromptTokens:     u.PromptTokens,
			CompletionTokens: u.CompletionTokens,
			TotalTokens:      u.TotalTokens,
		}
		if u.PromptTokensDetails.CachedTokens > 0 {
			usage.PromptTokensDetails = &model.PromptTokensDetails{
				CachedTokens: u.PromptTokensDetails.CachedTokens,
			}
		}
		if u.CompletionTokensDetails.ReasoningTokens > 0 {
			usage.CompletionTokensDetails = &model.CompletionTokensDetails{
				ReasoningTokens: u.CompletionTokensDetails.ReasoningTokens,
			}
		}
		result.Usage = usage
	}

	if raw, err := json.Marshal(result); err == nil {
		result.RawResponse = raw
		result.RawResponseFormat = model.APIFormatOpenAIChatCompletion
	}
	return result
}
