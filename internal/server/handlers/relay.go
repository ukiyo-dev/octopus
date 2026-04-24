package handlers

import (
	"net/http"

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

// dispatch routes requests based on path, falling back to passthrough for unknown endpoints.
func dispatch(c *gin.Context) {
	switch c.Param("path") {
	case "/chat/completions":
		relay.Handler(inbound.InboundTypeOpenAIChat, c)
	case "/responses":
		relay.Handler(inbound.InboundTypeOpenAIResponse, c)
	case "/messages":
		relay.Handler(inbound.InboundTypeAnthropic, c)
	case "/embeddings":
		relay.Handler(inbound.InboundTypeOpenAIEmbedding, c)
	default:
		relay.Handler(inbound.InboundTypePassthrough, c)
	}
}

func dispatchGemini(c *gin.Context) {
	relay.Handler(inbound.InboundTypeGemini, c)
}
