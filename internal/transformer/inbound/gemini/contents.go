package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/bestruirui/octopus/internal/transformer/model"
	"github.com/samber/lo"
)

var errInvalidGeminiPath = errors.New("invalid gemini request path")

type ContentsInbound struct {
	streamChunks   []*model.InternalLLMResponse
	storedResponse *model.InternalLLMResponse
}

type requestEnvelope struct {
	model.GeminiGenerateContentRequest
	SystemInstructionCamel *model.GeminiContent `json:"systemInstruction,omitempty"`
}

func (i *ContentsInbound) Probe(ctx context.Context, body []byte) (*model.ProbedRequest, error) {
	requestPath, ok := model.RequestPathFromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("%w: missing request path", errInvalidGeminiPath)
	}

	modelName, stream, err := extractRequestParams(requestPath)
	if err != nil {
		return nil, err
	}

	return &model.ProbedRequest{
		RawRequest:    body,
		InboundFormat: model.APIFormatGeminiContents,
		Model:         modelName,
		Stream:        stream,
		RequestKind:   model.RequestKindChat,
	}, nil
}

func (i *ContentsInbound) Parse(ctx context.Context, body []byte) (*model.InternalLLMRequest, error) {
	requestPath, ok := model.RequestPathFromContext(ctx)
	if !ok {
		return nil, fmt.Errorf("%w: missing request path", errInvalidGeminiPath)
	}

	modelName, stream, err := extractRequestParams(requestPath)
	if err != nil {
		return nil, err
	}

	var geminiReq requestEnvelope
	if err := json.Unmarshal(body, &geminiReq); err != nil {
		return nil, fmt.Errorf("failed to decode gemini request: %w", err)
	}
	if geminiReq.SystemInstruction == nil && geminiReq.SystemInstructionCamel != nil {
		geminiReq.SystemInstruction = geminiReq.SystemInstructionCamel
	}
	if len(geminiReq.Contents) == 0 {
		return nil, fmt.Errorf("contents are required")
	}

	req, err := convertGeminiToInternalRequest(&geminiReq.GeminiGenerateContentRequest)
	if err != nil {
		return nil, err
	}
	req.Model = modelName
	req.Stream = lo.ToPtr(stream)
	req.RawRequest = body
	req.RawAPIFormat = model.APIFormatGeminiContents
	return req, nil
}

func (i *ContentsInbound) TransformResponse(ctx context.Context, response *model.InternalLLMResponse) ([]byte, error) {
	i.storedResponse = response
	if response.RawResponseFormat == model.APIFormatGeminiContents && len(response.RawResponse) > 0 {
		return response.RawResponse, nil
	}
	body, err := json.Marshal(convertLLMToGeminiResponse(response, false))
	if err != nil {
		return nil, err
	}
	return body, nil
}

func (i *ContentsInbound) TransformStream(ctx context.Context, stream *model.InternalLLMResponse) ([]byte, error) {
	if stream == nil || stream.Object == "[DONE]" {
		return nil, nil
	}

	i.streamChunks = append(i.streamChunks, stream)

	if stream.RawResponseFormat == model.APIFormatGeminiContents && len(stream.RawResponse) > 0 {
		return []byte("data: " + string(stream.RawResponse) + "\n\n"), nil
	}

	body, err := json.Marshal(convertLLMToGeminiResponse(stream, true))
	if err != nil {
		return nil, err
	}
	return []byte("data: " + string(body) + "\n\n"), nil
}

func (i *ContentsInbound) GetInternalResponse(ctx context.Context) (*model.InternalLLMResponse, error) {
	if i.storedResponse != nil {
		if i.storedResponse.ResponseStatus == "" {
			i.storedResponse.ResponseStatus = model.ResponseStatusComplete
		}
		return i.storedResponse, nil
	}
	if len(i.streamChunks) == 0 {
		return nil, nil
	}

	result := aggregateStreamChunks(i.streamChunks)
	raw, err := json.Marshal(convertLLMToGeminiResponse(result, false))
	if err == nil {
		result.RawResponse = raw
		result.RawResponseFormat = model.APIFormatGeminiContents
	}
	i.streamChunks = nil
	return result, nil
}

func extractRequestParams(path string) (string, bool, error) {
	idx := strings.LastIndex(path, "/models/")
	if idx < 0 {
		return "", false, fmt.Errorf("%w: invalid request path %q", errInvalidGeminiPath, path)
	}

	suffix := path[idx+len("/models/"):]
	suffix = strings.TrimPrefix(suffix, "/")
	parts := strings.SplitN(suffix, ":", 2)
	if len(parts) != 2 || parts[0] == "" {
		return "", false, fmt.Errorf("%w: invalid request path %q", errInvalidGeminiPath, path)
	}

	switch parts[1] {
	case "generateContent":
		return parts[0], false, nil
	case "streamGenerateContent":
		return parts[0], true, nil
	default:
		return "", false, fmt.Errorf("%w: invalid request path %q", errInvalidGeminiPath, path)
	}
}

func convertGeminiToInternalRequest(geminiReq *model.GeminiGenerateContentRequest) (*model.InternalLLMRequest, error) {
	req := &model.InternalLLMRequest{
		TransformerMetadata: map[string]string{},
		RawAPIFormat:        model.APIFormatGeminiContents,
	}

	if geminiReq.GenerationConfig != nil {
		cfg := geminiReq.GenerationConfig
		if cfg.MaxOutputTokens > 0 {
			v := int64(cfg.MaxOutputTokens)
			req.MaxTokens = &v
		}
		req.Temperature = cfg.Temperature
		req.TopP = cfg.TopP
		if cfg.TopK != nil {
			req.TransformerMetadata["gemini_top_k"] = fmt.Sprintf("%d", *cfg.TopK)
		}
		if len(cfg.StopSequences) > 0 {
			if len(cfg.StopSequences) == 1 {
				req.Stop = &model.Stop{Stop: &cfg.StopSequences[0]}
			} else {
				req.Stop = &model.Stop{MultipleStop: append([]string(nil), cfg.StopSequences...)}
			}
		}
		if cfg.ThinkingConfig != nil {
			if cfg.ThinkingConfig.ThinkingLevel != "" {
				req.ReasoningEffort = strings.ToLower(cfg.ThinkingConfig.ThinkingLevel)
				if req.ReasoningEffort == "minimal" {
					req.ReasoningEffort = "low"
				}
			} else if cfg.ThinkingConfig.ThinkingBudget != nil {
				req.ReasoningEffort = thinkingBudgetToReasoningEffort(*cfg.ThinkingConfig.ThinkingBudget)
				budget := int64(*cfg.ThinkingConfig.ThinkingBudget)
				req.ReasoningBudget = &budget
			}
		}
		if len(cfg.ResponseModalities) > 0 {
			req.Modalities = convertGeminiModalitiesToLLM(cfg.ResponseModalities)
		}
		switch cfg.ResponseMimeType {
		case "application/json":
			req.ResponseFormat = &model.ResponseFormat{Type: "json_object"}
		case "text/plain":
			req.ResponseFormat = &model.ResponseFormat{Type: "text"}
		}
	}

	if len(geminiReq.SafetySettings) > 0 {
		if data, err := json.Marshal(geminiReq.SafetySettings); err == nil {
			req.TransformerMetadata["gemini_safety_settings"] = string(data)
		}
	}

	if geminiReq.SystemInstruction != nil {
		if text := extractTextFromContent(geminiReq.SystemInstruction); text != "" {
			req.Messages = append(req.Messages, model.Message{
				Role: "system",
				Content: model.MessageContent{
					Content: lo.ToPtr(text),
				},
			})
		}
	}

	for _, content := range geminiReq.Contents {
		messages, err := convertGeminiContentToMessages(content)
		if err != nil {
			return nil, err
		}
		req.Messages = append(req.Messages, messages...)
	}

	if len(geminiReq.Tools) > 0 {
		for _, tool := range geminiReq.Tools {
			for _, fd := range tool.FunctionDeclarations {
				if fd == nil {
					continue
				}
				params, _ := json.Marshal(fd.Parameters)
				req.Tools = append(req.Tools, model.Tool{
					Type: "function",
					Function: model.Function{
						Name:        fd.Name,
						Description: fd.Description,
						Parameters:  params,
					},
				})
			}
		}
	}

	if geminiReq.ToolConfig != nil && geminiReq.ToolConfig.FunctionCallingConfig != nil {
		fcc := geminiReq.ToolConfig.FunctionCallingConfig
		switch strings.ToUpper(fcc.Mode) {
		case "NONE":
			choice := "none"
			req.ToolChoice = &model.ToolChoice{ToolChoice: &choice}
		case "ANY":
			if len(fcc.AllowedFunctionNames) == 1 {
				req.ToolChoice = &model.ToolChoice{
					NamedToolChoice: &model.NamedToolChoice{
						Type:     "function",
						Function: model.ToolFunction{Name: fcc.AllowedFunctionNames[0]},
					},
				}
			} else {
				choice := "required"
				req.ToolChoice = &model.ToolChoice{ToolChoice: &choice}
			}
		default:
			choice := "auto"
			req.ToolChoice = &model.ToolChoice{ToolChoice: &choice}
		}
	}

	return req, nil
}

func convertGeminiContentToMessages(content *model.GeminiContent) ([]model.Message, error) {
	if content == nil {
		return nil, nil
	}

	role := content.Role
	switch role {
	case "", "user":
		return convertGeminiUserContent(content)
	case "model":
		return convertGeminiModelContent(content), nil
	default:
		return nil, fmt.Errorf("unsupported gemini role %q", role)
	}
}

func convertGeminiUserContent(content *model.GeminiContent) ([]model.Message, error) {
	msg := model.Message{Role: "user"}
	var parts []model.MessageContentPart
	var toolMessages []model.Message

	for _, part := range content.Parts {
		switch {
		case part == nil:
			continue
		case part.Text != "":
			text := part.Text
			parts = append(parts, model.MessageContentPart{Type: "text", Text: &text})
		case part.InlineData != nil:
			url := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MimeType, part.InlineData.Data)
			parts = append(parts, model.MessageContentPart{
				Type: "image_url",
				ImageURL: &model.ImageURL{
					URL: url,
				},
			})
		case part.FileData != nil:
			parts = append(parts, model.MessageContentPart{
				Type: "file",
				File: &model.File{
					Filename: part.FileData.FileURI,
					FileData: part.FileData.FileURI,
				},
			})
		case part.FunctionResponse != nil:
			payload, _ := json.Marshal(part.FunctionResponse.Response)
			text := string(payload)
			toolMessages = append(toolMessages, model.Message{
				Role:       "tool",
				ToolCallID: lo.ToPtr(part.FunctionResponse.Name),
				Content: model.MessageContent{
					Content: &text,
				},
			})
		}
	}

	var result []model.Message
	if len(parts) > 0 {
		if len(parts) == 1 && parts[0].Type == "text" {
			msg.Content = model.MessageContent{Content: parts[0].Text}
		} else {
			msg.Content = model.MessageContent{MultipleContent: parts}
		}
		result = append(result, msg)
	}
	result = append(result, toolMessages...)
	return result, nil
}

func convertGeminiModelContent(content *model.GeminiContent) []model.Message {
	msg := model.Message{Role: "assistant"}
	var textParts []string
	var multi []model.MessageContentPart
	var toolCalls []model.ToolCall
	var reasoning strings.Builder

	for idx, part := range content.Parts {
		switch {
		case part == nil:
			continue
		case part.Thought && part.Text != "":
			reasoning.WriteString(part.Text)
		case part.Text != "":
			textParts = append(textParts, part.Text)
			text := part.Text
			multi = append(multi, model.MessageContentPart{Type: "text", Text: &text})
		case part.InlineData != nil:
			url := fmt.Sprintf("data:%s;base64,%s", part.InlineData.MimeType, part.InlineData.Data)
			multi = append(multi, model.MessageContentPart{
				Type: "image_url",
				ImageURL: &model.ImageURL{
					URL: url,
				},
			})
		case part.FunctionCall != nil:
			args, _ := json.Marshal(part.FunctionCall.Args)
			toolCalls = append(toolCalls, model.ToolCall{
				Index: idx,
				ID:    fmt.Sprintf("call_%s_%d", part.FunctionCall.Name, idx),
				Type:  "function",
				Function: model.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				},
			})
		}
	}

	if len(multi) == 1 && multi[0].Type == "text" {
		msg.Content = model.MessageContent{Content: multi[0].Text}
	} else if len(multi) > 0 {
		msg.Content = model.MessageContent{MultipleContent: multi}
	} else if len(textParts) > 0 {
		text := strings.Join(textParts, "")
		msg.Content = model.MessageContent{Content: &text}
	}
	if reasoning.Len() > 0 {
		s := reasoning.String()
		msg.ReasoningContent = &s
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	return []model.Message{msg}
}

func extractTextFromContent(content *model.GeminiContent) string {
	if content == nil {
		return ""
	}
	var text strings.Builder
	for _, part := range content.Parts {
		if part != nil && part.Text != "" {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

func convertLLMToGeminiResponse(resp *model.InternalLLMResponse, isStream bool) *model.GeminiGenerateContentResponse {
	if resp == nil {
		return &model.GeminiGenerateContentResponse{}
	}

	result := &model.GeminiGenerateContentResponse{
		ModelVersion:  resp.Model,
		UsageMetadata: convertUsageToGemini(resp.Usage),
	}

	for _, choice := range resp.Choices {
		candidate := &model.GeminiCandidate{
			Index: choice.Index,
		}

		var msg *model.Message
		if isStream && choice.Delta != nil {
			msg = choice.Delta
		} else if choice.Message != nil {
			msg = choice.Message
		} else {
			msg = choice.Delta
		}

		if msg != nil {
			content := &model.GeminiContent{
				Role: "model",
			}
			if rc := msg.GetReasoningContent(); rc != "" {
				content.Parts = append(content.Parts, &model.GeminiPart{
					Text:    rc,
					Thought: true,
				})
			}
			if msg.Content.Content != nil && *msg.Content.Content != "" {
				content.Parts = append(content.Parts, &model.GeminiPart{
					Text: *msg.Content.Content,
				})
			}
			for _, part := range msg.Content.MultipleContent {
				switch part.Type {
				case "text":
					if part.Text != nil {
						content.Parts = append(content.Parts, &model.GeminiPart{Text: *part.Text})
					}
				case "image_url":
					if part.ImageURL != nil {
						if blob := parseDataURL(part.ImageURL.URL); blob != nil {
							content.Parts = append(content.Parts, &model.GeminiPart{InlineData: blob})
						}
					}
				}
			}
			for _, tc := range msg.ToolCalls {
				var args map[string]interface{}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				content.Parts = append(content.Parts, &model.GeminiPart{
					FunctionCall: &model.GeminiFunctionCall{
						Name: tc.Function.Name,
						Args: args,
					},
				})
			}
			candidate.Content = content
		}

		if choice.FinishReason != nil {
			reason := convertFinishReason(*choice.FinishReason)
			candidate.FinishReason = &reason
		}

		result.Candidates = append(result.Candidates, candidate)
	}

	return result
}

func aggregateStreamChunks(chunks []*model.InternalLLMResponse) *model.InternalLLMResponse {
	first := chunks[0]
	result := &model.InternalLLMResponse{
		ID:                first.ID,
		Object:            "chat.completion",
		Created:           first.Created,
		Model:             first.Model,
		SystemFingerprint: first.SystemFingerprint,
		ServiceTier:       first.ServiceTier,
	}

	choicesMap := make(map[int]*model.Choice)
	for _, chunk := range chunks {
		if chunk.ID != "" {
			result.ID = chunk.ID
		}
		if chunk.Model != "" {
			result.Model = chunk.Model
		}
		if chunk.Usage != nil {
			result.Usage = chunk.Usage
		}

		for _, choice := range chunk.Choices {
			existing, ok := choicesMap[choice.Index]
			if !ok {
				existing = &model.Choice{
					Index:   choice.Index,
					Message: &model.Message{},
				}
				choicesMap[choice.Index] = existing
			}

			src := choice.Delta
			if src == nil {
				src = choice.Message
			}
			if src != nil {
				if src.Role != "" {
					existing.Message.Role = src.Role
				}
				if src.Content.Content != nil {
					if existing.Message.Content.Content == nil {
						existing.Message.Content.Content = new(string)
					}
					*existing.Message.Content.Content += *src.Content.Content
				}
				if len(src.Content.MultipleContent) > 0 {
					existing.Message.Content.MultipleContent = append(existing.Message.Content.MultipleContent, src.Content.MultipleContent...)
				}
				if rc := src.GetReasoningContent(); rc != "" {
					if existing.Message.ReasoningContent == nil {
						existing.Message.ReasoningContent = new(string)
					}
					*existing.Message.ReasoningContent += rc
				}
				for _, tc := range src.ToolCalls {
					existing.Message.ToolCalls = mergeToolCall(existing.Message.ToolCalls, tc)
				}
			}

			if choice.FinishReason != nil {
				existing.FinishReason = choice.FinishReason
			}
		}
	}

	for idx := 0; idx < len(choicesMap); idx++ {
		if choice, ok := choicesMap[idx]; ok {
			result.Choices = append(result.Choices, *choice)
		}
	}
	if len(result.Choices) == 0 {
		for _, choice := range choicesMap {
			result.Choices = append(result.Choices, *choice)
		}
	}

	return result
}

func mergeToolCall(toolCalls []model.ToolCall, delta model.ToolCall) []model.ToolCall {
	for idx, tc := range toolCalls {
		if tc.Index != delta.Index {
			continue
		}
		if delta.ID != "" {
			toolCalls[idx].ID = delta.ID
		}
		if delta.Type != "" {
			toolCalls[idx].Type = delta.Type
		}
		if delta.Function.Name != "" {
			toolCalls[idx].Function.Name = delta.Function.Name
		}
		if delta.Function.Arguments != "" {
			toolCalls[idx].Function.Arguments += delta.Function.Arguments
		}
		return toolCalls
	}
	return append(toolCalls, delta)
}

func convertUsageToGemini(usage *model.Usage) *model.GeminiUsageMetadata {
	if usage == nil {
		return nil
	}
	return &model.GeminiUsageMetadata{
		PromptTokenCount:     int(usage.PromptTokens),
		CandidatesTokenCount: int(usage.CompletionTokens),
		TotalTokenCount:      int(usage.TotalTokens),
		CachedContentTokenCount: func() int {
			if usage.PromptTokensDetails == nil {
				return 0
			}
			return int(usage.PromptTokensDetails.CachedTokens)
		}(),
	}
}

func convertFinishReason(reason string) string {
	switch reason {
	case "length":
		return "MAX_TOKENS"
	case "content_filter":
		return "SAFETY"
	default:
		return "STOP"
	}
}

func thinkingBudgetToReasoningEffort(budget int32) string {
	switch {
	case budget <= 0:
		return ""
	case budget <= 1024:
		return "low"
	case budget <= 4096:
		return "medium"
	default:
		return "high"
	}
}

func convertGeminiModalitiesToLLM(modalities []string) []string {
	result := make([]string, 0, len(modalities))
	for _, modality := range modalities {
		switch strings.ToUpper(modality) {
		case "TEXT":
			result = append(result, "text")
		case "IMAGE":
			result = append(result, "image")
		case "AUDIO":
			result = append(result, "audio")
		}
	}
	return result
}

func parseDataURL(raw string) *model.GeminiBlob {
	if !strings.HasPrefix(raw, "data:") {
		return nil
	}
	parts := strings.SplitN(strings.TrimPrefix(raw, "data:"), ";base64,", 2)
	if len(parts) != 2 {
		return nil
	}
	return &model.GeminiBlob{
		MimeType: parts[0],
		Data:     parts[1],
	}
}
