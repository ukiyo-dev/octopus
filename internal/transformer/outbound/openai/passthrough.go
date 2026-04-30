package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
)

// parseSSEDataLines extracts the payload from each "data: ..." line in raw SSE bytes.
// Uses a 32 MB scanner buffer so that large response.completed events are not dropped.
func parseSSEDataLines(rawSSE []byte) [][]byte {
	var out [][]byte
	scanner := bufio.NewScanner(bytes.NewReader(rawSSE))
	scanner.Buffer(make([]byte, 64*1024), 32*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			out = append(out, []byte(strings.TrimPrefix(line, "data: ")))
		}
	}
	return out
}

// patchRawRequest applies targeted patches to a raw JSON request body without full deserialization:
//   - replaces the "model" field when targetModel differs from the body's value (enables model aliasing)
//   - injects stream_options.include_usage=true when patchStream=true and stream=true (Chat API only)
func patchRawRequest(body []byte, targetModel string, patchStream bool) []byte {
	var minimal struct {
		Model         string          `json:"model"`
		Stream        *bool           `json:"stream"`
		StreamOptions json.RawMessage `json:"stream_options"`
	}
	if err := json.Unmarshal(body, &minimal); err != nil {
		return body
	}

	needModelPatch := targetModel != "" && minimal.Model != targetModel

	needStreamPatch := patchStream && minimal.Stream != nil && *minimal.Stream
	if needStreamPatch && minimal.StreamOptions != nil {
		var opts struct {
			IncludeUsage bool `json:"include_usage"`
		}
		if err := json.Unmarshal(minimal.StreamOptions, &opts); err == nil && opts.IncludeUsage {
			needStreamPatch = false
		}
	}

	if !needModelPatch && !needStreamPatch {
		return body
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}

	if needModelPatch {
		if patched, err := json.Marshal(targetModel); err == nil {
			m["model"] = patched
		}
	}

	if needStreamPatch {
		if existing, ok := m["stream_options"]; ok {
			var opts map[string]json.RawMessage
			if err := json.Unmarshal(existing, &opts); err == nil {
				opts["include_usage"] = json.RawMessage("true")
				if patched, err := json.Marshal(opts); err == nil {
					m["stream_options"] = patched
				}
			}
		} else {
			m["stream_options"] = json.RawMessage(`{"include_usage":true}`)
		}
	}

	if patched, err := json.Marshal(m); err == nil {
		return patched
	}
	return body
}
