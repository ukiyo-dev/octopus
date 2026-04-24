package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"strings"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

// PassthroughInbound handles unknown or future API endpoints by forwarding raw bytes.
// It only extracts model and stream from the request; all other fields are carried in RawRequest.
type PassthroughInbound struct {
	storedResponse *model.InternalLLMResponse
	streamChunks   []*model.InternalLLMResponse
	rawStreamBytes []byte // collected by handleRawStream for post-stream usage extraction
}

func (i *PassthroughInbound) TransformRequest(ctx context.Context, body []byte) (*model.InternalLLMRequest, error) {
	var minimal struct {
		Model  string `json:"model"`
		Stream *bool  `json:"stream"`
	}
	// Best-effort parse: non-JSON bodies (multipart etc.) are allowed; model may be empty.
	json.Unmarshal(body, &minimal) //nolint:errcheck
	return &model.InternalLLMRequest{
		Model:        minimal.Model,
		Stream:       minimal.Stream,
		RawRequest:   body,
		RawAPIFormat: model.APIFormatPassthrough,
	}, nil
}

func (i *PassthroughInbound) TransformResponse(ctx context.Context, response *model.InternalLLMResponse) ([]byte, error) {
	i.storedResponse = response
	if len(response.RawResponse) > 0 {
		return response.RawResponse, nil
	}
	return json.Marshal(response)
}

func (i *PassthroughInbound) TransformStream(ctx context.Context, stream *model.InternalLLMResponse) ([]byte, error) {
	if stream.Object == "[DONE]" {
		return []byte("data: [DONE]\n\n"), nil
	}
	i.streamChunks = append(i.streamChunks, stream)
	if len(stream.RawResponse) > 0 {
		return []byte("data: " + string(stream.RawResponse) + "\n\n"), nil
	}
	body, err := json.Marshal(stream)
	if err != nil {
		return nil, err
	}
	return []byte("data: " + string(body) + "\n\n"), nil
}

// CollectRawStream is called by handleRawStream after the stream ends.
// It stores the full raw SSE bytes for post-stream usage extraction.
func (i *PassthroughInbound) CollectRawStream(data []byte) {
	i.rawStreamBytes = data
}

func (i *PassthroughInbound) GetInternalResponse(ctx context.Context) (*model.InternalLLMResponse, error) {
	if i.storedResponse != nil {
		return i.storedResponse, nil
	}
	if len(i.rawStreamBytes) > 0 {
		result := parseUsageFromSSEBytes(i.rawStreamBytes)
		result.RawResponse = i.rawStreamBytes
		return result, nil
	}
	if len(i.streamChunks) == 0 {
		return nil, nil
	}
	result := &model.InternalLLMResponse{}
	for _, chunk := range i.streamChunks {
		if chunk.Usage != nil {
			result.Usage = chunk.Usage
		}
		if chunk.Model != "" {
			result.Model = chunk.Model
		}
		if chunk.ID != "" {
			result.ID = chunk.ID
		}
	}
	i.streamChunks = nil
	return result, nil
}

// parseUsageFromSSEBytes scans raw SSE bytes and extracts the last seen id, model, and usage.
// It handles both Chat Completions format (prompt_tokens/completion_tokens) and
// Responses API format (input_tokens/output_tokens, either top-level or nested under "response").
func parseUsageFromSSEBytes(data []byte) *model.InternalLLMResponse {
	result := &model.InternalLLMResponse{}

	var ev struct {
		ID    string `json:"id"`
		Model string `json:"model"`
		Usage *struct {
			// Chat Completions
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
			TotalTokens      int64 `json:"total_tokens"`
			// Responses API (top-level or embedded)
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
		// Responses API wraps metadata under "response"
		Response *struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage *struct {
				InputTokens  int64 `json:"input_tokens"`
				OutputTokens int64 `json:"output_tokens"`
				TotalTokens  int64 `json:"total_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			continue
		}

		if ev.ID != "" {
			result.ID = ev.ID
		}
		if ev.Model != "" {
			result.Model = ev.Model
		}

		// Chat Completions usage
		if ev.Usage != nil {
			u := &model.Usage{}
			if ev.Usage.PromptTokens > 0 || ev.Usage.CompletionTokens > 0 {
				u.PromptTokens = ev.Usage.PromptTokens
				u.CompletionTokens = ev.Usage.CompletionTokens
				u.TotalTokens = ev.Usage.TotalTokens
			} else if ev.Usage.InputTokens > 0 || ev.Usage.OutputTokens > 0 {
				u.PromptTokens = ev.Usage.InputTokens
				u.CompletionTokens = ev.Usage.OutputTokens
				u.TotalTokens = ev.Usage.TotalTokens
				if u.TotalTokens == 0 {
					u.TotalTokens = u.PromptTokens + u.CompletionTokens
				}
			}
			if u.PromptTokens > 0 || u.CompletionTokens > 0 {
				result.Usage = u
			}
		}

		// Responses API nested usage
		if ev.Response != nil {
			if ev.Response.ID != "" {
				result.ID = ev.Response.ID
			}
			if ev.Response.Model != "" {
				result.Model = ev.Response.Model
			}
			if ev.Response.Usage != nil {
				u := &model.Usage{
					PromptTokens:     ev.Response.Usage.InputTokens,
					CompletionTokens: ev.Response.Usage.OutputTokens,
					TotalTokens:      ev.Response.Usage.TotalTokens,
				}
				if u.TotalTokens == 0 {
					u.TotalTokens = u.PromptTokens + u.CompletionTokens
				}
				if u.PromptTokens > 0 || u.CompletionTokens > 0 {
					result.Usage = u
				}
			}
		}
	}

	return result
}
