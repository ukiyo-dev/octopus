package openai

import (
	"context"
	"encoding/json"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

// PassthroughInbound handles unknown or future API endpoints by forwarding raw bytes.
// It only extracts model and stream from the request; all other fields are carried in RawRequest.
type PassthroughInbound struct {
	storedResponse *model.InternalLLMResponse
	streamChunks   []*model.InternalLLMResponse
}

func (i *PassthroughInbound) Probe(ctx context.Context, body []byte) (*model.ProbedRequest, error) {
	var minimal struct {
		Model  string `json:"model"`
		Stream *bool  `json:"stream"`
	}
	// Best-effort parse: non-JSON bodies (multipart etc.) are allowed; model may be empty.
	json.Unmarshal(body, &minimal) //nolint:errcheck
	return &model.ProbedRequest{
		RawRequest:    body,
		InboundFormat: model.APIFormatPassthrough,
		Model:         minimal.Model,
		Stream:        minimal.Stream != nil && *minimal.Stream,
		RequestKind:   model.RequestKindPassthrough,
	}, nil
}

func (i *PassthroughInbound) Parse(ctx context.Context, body []byte) (*model.InternalLLMRequest, error) {
	var minimal struct {
		Model  string `json:"model"`
		Stream *bool  `json:"stream"`
	}
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

func (i *PassthroughInbound) GetInternalResponse(ctx context.Context) (*model.InternalLLMResponse, error) {
	if i.storedResponse != nil {
		return i.storedResponse, nil
	}
	if len(i.streamChunks) == 0 {
		return nil, nil
	}
	result := &model.InternalLLMResponse{}
	events := make([]json.RawMessage, 0, len(i.streamChunks))
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
		if len(chunk.RawResponse) > 0 && json.Valid(chunk.RawResponse) {
			events = append(events, json.RawMessage(chunk.RawResponse))
		}
	}
	if len(events) > 0 {
		if raw, err := json.Marshal(map[string]any{"events": events}); err == nil {
			result.RawResponse = raw
			result.RawResponseFormat = model.APIFormatPassthrough
		}
	}
	i.streamChunks = nil
	return result, nil
}

