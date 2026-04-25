package handlers

import (
	"net/http"
	"strings"

	"github.com/bestruirui/octopus/internal/relay"
	"github.com/bestruirui/octopus/internal/server/middleware"
	"github.com/bestruirui/octopus/internal/server/router"
	"github.com/bestruirui/octopus/internal/transformer/inbound"
	"github.com/gin-gonic/gin"
)

func init() {
	router.NewGroupRouter("/v1").
		Use(middleware.APIKeyAuth()).
		Use(middleware.RequireJSON()).
		AddRoute(
			router.NewRoute("/*path", http.MethodPost).
				Handle(dispatch),
		)
	router.NewGroupRouter("/v1beta").
		Use(middleware.APIKeyAuth()).
		Use(middleware.RequireJSON()).
		AddRoute(
			router.NewRoute("/models/*path", http.MethodPost).
				Handle(dispatchGemini),
		)
}

// dispatch routes requests based on path. Sub-paths of known protocol endpoints
// are routed to their parent protocol's inbound adapter (as sidecar passthrough).
func dispatch(c *gin.Context) {
	path := c.Param("path")
	switch {
	case path == "/chat/completions":
		relay.Handler(inbound.InboundTypeOpenAIChat, c)
	case path == "/responses" || strings.HasPrefix(path, "/responses/"):
		relay.Handler(inbound.InboundTypeOpenAIResponse, c)
	case path == "/messages" || strings.HasPrefix(path, "/messages/"):
		relay.Handler(inbound.InboundTypeAnthropic, c)
	case path == "/embeddings":
		relay.Handler(inbound.InboundTypeOpenAIEmbedding, c)
	default:
		relay.Handler(inbound.InboundTypePassthrough, c)
	}
}

func dispatchGemini(c *gin.Context) {
	relay.Handler(inbound.InboundTypeGemini, c)
}
