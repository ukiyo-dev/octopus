package resp

import (
	"net/http"

	"github.com/bestruirui/octopus/internal/utils/log"
	"github.com/gin-gonic/gin"
)

type ResponseStruct struct {
	Code    int         `json:"code" example:"200"`
	Message string      `json:"message" example:"success"`
	Data    interface{} `json:"data,omitempty"`
}

func Success(c *gin.Context, data any) {
	c.JSON(http.StatusOK, ResponseStruct{
		Code:    http.StatusOK,
		Message: "success",
		Data:    data,
	})
}

func Error(c *gin.Context, code int, err string) {
	method := ""
	path := ""
	query := ""
	if c.Request != nil && c.Request.URL != nil {
		method = c.Request.Method
		path = c.Request.URL.Path
		query = c.Request.URL.RawQuery
	}
	log.Warnf("request failed: status=%d, method=%s, path=%s, query=%s, error=%s",
		code, method, path, query, err)
	c.AbortWithStatusJSON(code, ResponseStruct{
		Code:    code,
		Message: err,
	})
}
