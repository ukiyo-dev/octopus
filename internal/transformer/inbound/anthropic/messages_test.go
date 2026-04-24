package anthropic

import (
	"context"
	"strings"
	"testing"
)

func TestMessagesInboundParseRejectsDocumentContent(t *testing.T) {
	inbound := &MessagesInbound{}
	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":16,
		"messages":[
			{
				"role":"user",
				"content":[
					{"type":"document","source":{"type":"url","url":"https://example.com/doc.pdf"}}
				]
			}
		]
	}`)

	_, err := inbound.Parse(context.Background(), body)
	if err == nil {
		t.Fatal("expected Parse to reject document content")
	}
	if !strings.Contains(err.Error(), "same-protocol passthrough") {
		t.Fatalf("expected passthrough guidance, got %v", err)
	}
}
