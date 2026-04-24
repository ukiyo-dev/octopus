package gemini

import (
	"context"
	"testing"

	transformerModel "github.com/bestruirui/octopus/internal/transformer/model"
)

func TestExtractRequestParams(t *testing.T) {
	modelName, stream, err := extractRequestParams("/v1beta/models/gemini-2.5-pro:streamGenerateContent")
	if err != nil {
		t.Fatalf("extractRequestParams returned error: %v", err)
	}
	if modelName != "gemini-2.5-pro" {
		t.Fatalf("unexpected model name %q", modelName)
	}
	if !stream {
		t.Fatal("expected stream request")
	}
}

func TestContentsInboundParse(t *testing.T) {
	inbound := &ContentsInbound{}
	ctx := transformerModel.WithRequestPath(context.Background(), "/v1beta/models/gemini-2.5-pro:generateContent")
	body := []byte(`{
		"contents":[
			{"role":"user","parts":[{"text":"hello"}]}
		],
		"tools":[
			{"functionDeclarations":[{"name":"lookup","description":"Lookup data","parameters":{"type":"object"}}]}
		]
	}`)

	req, err := inbound.Parse(ctx, body)
	if err != nil {
		t.Fatalf("Parse returned error: %v", err)
	}
	if req.Model != "gemini-2.5-pro" {
		t.Fatalf("unexpected model %q", req.Model)
	}
	if req.Stream == nil || *req.Stream {
		t.Fatal("expected non-stream request")
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Fatalf("unexpected messages: %+v", req.Messages)
	}
	if len(req.Tools) != 1 || req.Tools[0].Function.Name != "lookup" {
		t.Fatalf("unexpected tools: %+v", req.Tools)
	}
}
