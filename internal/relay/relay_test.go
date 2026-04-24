package relay

import (
	"context"
	"net/http"
	"testing"

	dbmodel "github.com/bestruirui/octopus/internal/model"
	"github.com/bestruirui/octopus/internal/relay/balancer"
	transformerModel "github.com/bestruirui/octopus/internal/transformer/model"
)

type fakeInbound struct {
	parseCalls int
	parseReq   *transformerModel.InternalLLMRequest
	parseErr   error
}

func (f *fakeInbound) Probe(ctx context.Context, body []byte) (*transformerModel.ProbedRequest, error) {
	return nil, nil
}

func (f *fakeInbound) Parse(ctx context.Context, body []byte) (*transformerModel.InternalLLMRequest, error) {
	f.parseCalls++
	if f.parseErr != nil {
		return nil, f.parseErr
	}
	return f.parseReq, nil
}

func (f *fakeInbound) TransformResponse(ctx context.Context, response *transformerModel.InternalLLMResponse) ([]byte, error) {
	return nil, nil
}

func (f *fakeInbound) TransformStream(ctx context.Context, stream *transformerModel.InternalLLMResponse) ([]byte, error) {
	return nil, nil
}

func (f *fakeInbound) GetInternalResponse(ctx context.Context) (*transformerModel.InternalLLMResponse, error) {
	return nil, nil
}

type fakeOutbound struct {
	targetFormat transformerModel.APIFormat
}

func (f *fakeOutbound) TargetFormat() transformerModel.APIFormat {
	return f.targetFormat
}

func (f *fakeOutbound) TransformRequest(ctx context.Context, request *transformerModel.InternalLLMRequest, baseURL, key string) (*http.Request, error) {
	return nil, nil
}

func (f *fakeOutbound) TransformResponse(ctx context.Context, response *http.Response) (*transformerModel.InternalLLMResponse, error) {
	return nil, nil
}

func (f *fakeOutbound) TransformStream(ctx context.Context, eventData []byte) (*transformerModel.InternalLLMResponse, error) {
	return nil, nil
}

func newTestIterator(t *testing.T, modelName string) *balancer.Iterator {
	t.Helper()

	iter := balancer.NewIterator(dbmodel.Group{
		Mode: dbmodel.GroupModeRoundRobin,
		Items: []dbmodel.GroupItem{
			{
				ChannelID: 1,
				ModelName: modelName,
			},
		},
	}, 1, "request-model")

	if !iter.Next() {
		t.Fatal("expected iterator candidate")
	}
	return iter
}

func TestBuildInternalRequestSameFormatSkipsParse(t *testing.T) {
	inbound := &fakeInbound{}
	attempt := &relayAttempt{
		relayRequest: &relayRequest{
			inAdapter: inbound,
			probedRequest: &transformerModel.ProbedRequest{
				RawRequest:    []byte(`{"model":"source","stream":true}`),
				InboundFormat: transformerModel.APIFormatOpenAIChatCompletion,
				Model:         "source",
				Stream:        true,
				Query:         map[string][]string{"foo": {"bar"}},
				RawPath:       "/chat/completions",
				RequestKind:   transformerModel.RequestKindChat,
			},
			iter: newTestIterator(t, "target-model"),
		},
		outAdapter: &fakeOutbound{targetFormat: transformerModel.APIFormatOpenAIChatCompletion},
	}

	req, err := attempt.buildInternalRequest(context.Background())
	if err != nil {
		t.Fatalf("buildInternalRequest returned error: %v", err)
	}
	if inbound.parseCalls != 0 {
		t.Fatalf("expected Parse not to be called, got %d", inbound.parseCalls)
	}
	if req.Model != "target-model" {
		t.Fatalf("expected target model, got %q", req.Model)
	}
	if req.RawAPIFormat != transformerModel.APIFormatOpenAIChatCompletion {
		t.Fatalf("expected raw format passthrough, got %q", req.RawAPIFormat)
	}
	if string(req.RawRequest) != `{"model":"source","stream":true}` {
		t.Fatalf("expected raw request to be preserved, got %q", string(req.RawRequest))
	}
	if req.Stream == nil || !*req.Stream {
		t.Fatalf("expected stream flag to be preserved")
	}
	if req.RawPath != "/chat/completions" {
		t.Fatalf("expected raw path to be preserved, got %q", req.RawPath)
	}
	if got := req.Query.Get("foo"); got != "bar" {
		t.Fatalf("expected query to be preserved, got %q", got)
	}
}

func TestBuildInternalRequestCrossFormatParses(t *testing.T) {
	inbound := &fakeInbound{
		parseReq: &transformerModel.InternalLLMRequest{
			Model: "source",
			Messages: []transformerModel.Message{
				{
					Role: "user",
					Content: transformerModel.MessageContent{
						Content: strPtr("hello"),
					},
				},
			},
			RawAPIFormat: transformerModel.APIFormatOpenAIChatCompletion,
			RawRequest:   []byte(`{"model":"source","messages":[{"role":"user","content":"hello"}]}`),
		},
	}
	attempt := &relayAttempt{
		relayRequest: &relayRequest{
			inAdapter: inbound,
			probedRequest: &transformerModel.ProbedRequest{
				RawRequest:    []byte(`{"model":"source","messages":[{"role":"user","content":"hello"}]}`),
				InboundFormat: transformerModel.APIFormatOpenAIChatCompletion,
				Model:         "source",
				Query:         map[string][]string{"foo": {"bar"}},
				RawPath:       "/chat/completions",
				RequestKind:   transformerModel.RequestKindChat,
			},
			iter: newTestIterator(t, "target-model"),
		},
		outAdapter: &fakeOutbound{targetFormat: transformerModel.APIFormatAnthropicMessage},
	}

	req, err := attempt.buildInternalRequest(context.Background())
	if err != nil {
		t.Fatalf("buildInternalRequest returned error: %v", err)
	}
	if inbound.parseCalls != 1 {
		t.Fatalf("expected Parse to be called once, got %d", inbound.parseCalls)
	}
	if req.Model != "target-model" {
		t.Fatalf("expected parsed request model to be patched, got %q", req.Model)
	}
	if req.RawPath != "/chat/completions" {
		t.Fatalf("expected raw path to be preserved, got %q", req.RawPath)
	}
	if got := req.Query.Get("foo"); got != "bar" {
		t.Fatalf("expected query to be preserved, got %q", got)
	}
}

func strPtr(v string) *string {
	return &v
}
