package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

type EmbeddingOutbound struct{}

// OpenAIEmbeddingRequest 是 OpenAI 标准的请求格式（发送给上游）
type OpenAIEmbeddingRequest struct {
	Model          string               `json:"model"`
	Input          model.EmbeddingInput `json:"input"` // 上游期望 "input"
	Dimensions     *int64               `json:"dimensions,omitempty"`
	EncodingFormat *string              `json:"encoding_format,omitempty"`
	User           *string              `json:"user,omitempty"`
}

// OpenAIEmbeddingResponse 是 OpenAI 标准的响应格式（上游返回）
type OpenAIEmbeddingResponse struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Data    []model.EmbeddingObject `json:"data"` // 上游返回 "data"
	Usage   *model.Usage            `json:"usage,omitempty"`
}

func (o *EmbeddingOutbound) TransformRequest(ctx context.Context, request *model.InternalLLMRequest, baseUrl, key string) (*http.Request, error) {
	if !request.IsEmbeddingRequest() && request.RawAPIFormat != model.APIFormatPassthrough {
		return nil, errors.New("not an embedding request")
	}

	passthrough := len(request.RawRequest) > 0 && (request.RawAPIFormat == model.APIFormatOpenAIEmbedding || request.RawAPIFormat == model.APIFormatPassthrough)

	var body []byte
	var err error

	if passthrough {
		body = patchRawRequest(request.RawRequest, request.Model, false)
	} else {
		embeddingRequest := map[string]any{
			"model": request.Model,
			"input": request.EmbeddingInput,
		}
		if request.EmbeddingDimensions != nil {
			embeddingRequest["dimensions"] = *request.EmbeddingDimensions
		}
		if request.EmbeddingEncodingFormat != nil {
			embeddingRequest["encoding_format"] = *request.EmbeddingEncodingFormat
		}
		if request.User != nil {
			embeddingRequest["user"] = *request.User
		}
		body, err = json.Marshal(embeddingRequest)
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
		parsedUrl.Path = parsedUrl.Path + "/embeddings"
	}
	req.URL = parsedUrl
	req.Method = http.MethodPost
	return req, nil
}

func (o *EmbeddingOutbound) TransformResponse(ctx context.Context, response *http.Response) (*model.InternalLLMResponse, error) {
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("response body is empty")
	}

	// 先解析为 OpenAI 标准格式
	var openAIResp OpenAIEmbeddingResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	resp := &model.InternalLLMResponse{
		ID:                openAIResp.ID,
		Object:            openAIResp.Object,
		Created:           openAIResp.Created,
		Model:             openAIResp.Model,
		EmbeddingData:     openAIResp.Data,
		Usage:             openAIResp.Usage,
		RawResponse:       body,
		RawResponseFormat: model.APIFormatOpenAIEmbedding,
	}

	return resp, nil
}

func (o *EmbeddingOutbound) TransformStream(ctx context.Context, eventData []byte) (*model.InternalLLMResponse, error) {
	// Embedding API does not support streaming
	return nil, errors.New("streaming is not supported for embedding API")
}
