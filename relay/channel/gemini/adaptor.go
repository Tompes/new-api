package gemini

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"one-api/common"
	"one-api/dto"
	"one-api/relay/channel"
	relaycommon "one-api/relay/common"
	"one-api/relay/constant"
	"one-api/service"
	"one-api/setting/model_setting"
	"strings"

	"github.com/gin-gonic/gin"
)

type Adaptor struct {
}

func (a *Adaptor) ConvertClaudeRequest(c *gin.Context, info *relaycommon.RelayInfo, request *dto.ClaudeRequest) (any, error) {
	geminiRequest, err := ConvertClaudeRequest(request)
	if err != nil {
		return nil, err
	}
	return geminiRequest, nil
}

func (a *Adaptor) ConvertAudioRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.AudioRequest) (io.Reader, error) {
	format, base64String, err := service.DecodeBase64FileData(request.Input)
	if err != nil {
		return nil, fmt.Errorf("decode base64 audio data failed: %s", err.Error())
	}

	geminiRequest := GeminiChatRequest{
		Contents: []GeminiChatContent{
			{
				Role: "user",
				Parts: []GeminiPart{
					{
						InlineData: &GeminiInlineData{
							MimeType: "audio/" + format,
							Data:     base64String,
						},
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(geminiRequest)
	if err != nil {
		return nil, fmt.Errorf("error marshalling gemini audio request: %w", err)
	}

	return bytes.NewReader(jsonData), nil
}

func (a *Adaptor) ConvertImageRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.ImageRequest) (any, error) {
	if !strings.HasPrefix(info.UpstreamModelName, "imagen") {
		return nil, errors.New("not supported model for image generation")
	}

	// convert size to aspect ratio
	aspectRatio := "1:1" // default aspect ratio
	switch request.Size {
	case "1024x1024":
		aspectRatio = "1:1"
	case "1024x1792":
		aspectRatio = "9:16"
	case "1792x1024":
		aspectRatio = "16:9"
	}

	// build gemini imagen request
	geminiRequest := GeminiImageRequest{
		Instances: []GeminiImageInstance{
			{
				Prompt: request.Prompt,
			},
		},
		Parameters: GeminiImageParameters{
			SampleCount:      request.N,
			AspectRatio:      aspectRatio,
			PersonGeneration: "allow_adult", // default allow adult
		},
	}

	return geminiRequest, nil
}

func (a *Adaptor) Init(info *relaycommon.RelayInfo) {

}

func (a *Adaptor) GetRequestURL(info *relaycommon.RelayInfo) (string, error) {

	if model_setting.GetGeminiSettings().ThinkingAdapterEnabled {
		// 新增逻辑：处理 -thinking-<budget> 格式
		if strings.Contains(info.UpstreamModelName, "-thinking-") {
			parts := strings.Split(info.UpstreamModelName, "-thinking-")
			info.UpstreamModelName = parts[0]
		} else if strings.HasSuffix(info.UpstreamModelName, "-thinking") { // 旧的适配
			info.UpstreamModelName = strings.TrimSuffix(info.UpstreamModelName, "-thinking")
		} else if strings.HasSuffix(info.UpstreamModelName, "-nothinking") {
			info.UpstreamModelName = strings.TrimSuffix(info.UpstreamModelName, "-nothinking")
		}
	}

	version := model_setting.GetGeminiVersionSetting(info.UpstreamModelName)

	if strings.HasPrefix(info.UpstreamModelName, "imagen") {
		return fmt.Sprintf("%s/%s/models/%s:predict", info.BaseUrl, version, info.UpstreamModelName), nil
	}

	if strings.HasPrefix(info.UpstreamModelName, "text-embedding") ||
		strings.HasPrefix(info.UpstreamModelName, "embedding") ||
		strings.HasPrefix(info.UpstreamModelName, "gemini-embedding") {
		return fmt.Sprintf("%s/%s/models/%s:embedContent", info.BaseUrl, version, info.UpstreamModelName), nil
	}

	action := "generateContent"
	if info.IsStream {
		action = "streamGenerateContent?alt=sse"
	}
	return fmt.Sprintf("%s/%s/models/%s:%s", info.BaseUrl, version, info.UpstreamModelName, action), nil
}

func (a *Adaptor) SetupRequestHeader(c *gin.Context, req *http.Header, info *relaycommon.RelayInfo) error {
	channel.SetupApiRequestHeader(info, c, req)
	req.Set("x-goog-api-key", info.ApiKey)
	return nil
}

func (a *Adaptor) ConvertOpenAIRequest(c *gin.Context, info *relaycommon.RelayInfo, request *dto.GeneralOpenAIRequest) (any, error) {
	if request == nil {
		return nil, errors.New("request is nil")
	}

	geminiRequest, err := CovertGemini2OpenAI(*request, info)
	if err != nil {
		return nil, err
	}

	return geminiRequest, nil
}

func (a *Adaptor) ConvertRerankRequest(c *gin.Context, relayMode int, request dto.RerankRequest) (any, error) {
	return nil, nil
}

func (a *Adaptor) ConvertEmbeddingRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.EmbeddingRequest) (any, error) {
	if request.Input == nil {
		return nil, errors.New("input is required")
	}

	inputs := request.ParseInput()
	if len(inputs) == 0 {
		return nil, errors.New("input is empty")
	}

	// only process the first input
	geminiRequest := GeminiEmbeddingRequest{
		Content: GeminiChatContent{
			Parts: []GeminiPart{
				{
					Text: inputs[0],
				},
			},
		},
	}

	// set specific parameters for different models
	// https://ai.google.dev/api/embeddings?hl=zh-cn#method:-models.embedcontent
	switch info.UpstreamModelName {
	case "text-embedding-004":
		// except embedding-001 supports setting `OutputDimensionality`
		if request.Dimensions > 0 {
			geminiRequest.OutputDimensionality = request.Dimensions
		}
	}

	return geminiRequest, nil
}

func (a *Adaptor) ConvertOpenAIResponsesRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.OpenAIResponsesRequest) (any, error) {
	geminiRequest := &GeminiChatRequest{
		Contents: []GeminiChatContent{},
		GenerationConfig: GeminiChatGenerationConfig{
			Temperature:     &request.Temperature,
			TopP:            request.TopP,
			MaxOutputTokens: request.MaxOutputTokens,
		},
	}

	if request.Instructions != nil {
		var instructions string
		if err := json.Unmarshal(request.Instructions, &instructions); err == nil {
			geminiRequest.SystemInstructions = &GeminiChatContent{
				Parts: []GeminiPart{
					{
						Text: instructions,
					},
				},
			}
		}
	}

	if request.Input != nil {
		var inputMessages []dto.Message
		if err := json.Unmarshal(request.Input, &inputMessages); err == nil {
			for _, message := range inputMessages {
				geminiContent, err := openAIMessageToGeminiContent(message)
				if err != nil {
					return nil, err
				}
				geminiRequest.Contents = append(geminiRequest.Contents, *geminiContent)
			}
		}
	}

	for _, tool := range request.Tools {
		if tool.Type == "web_search" {
			geminiRequest.Tools = append(geminiRequest.Tools, GeminiChatTool{
				GoogleSearch: make(map[string]string),
			})
		}
		// Simplified function tool handling
		if tool.Type == "function" && tool.Function != nil {
			var function dto.FunctionRequest
			if err := json.Unmarshal(tool.Function, &function); err == nil {
				geminiRequest.Tools = append(geminiRequest.Tools, GeminiChatTool{
					FunctionDeclarations: []dto.FunctionRequest{function},
				})
			}
		}
	}

	return geminiRequest, nil
}

func openAIMessageToGeminiContent(message dto.Message) (*GeminiChatContent, error) {
	role := message.Role
	if role == "assistant" {
		role = "model"
	}

	content := &GeminiChatContent{
		Role: role,
	}

	parts, err := openAIMessageContentToGeminiParts(message.Content)
	if err != nil {
		return nil, err
	}
	content.Parts = parts

	return content, nil
}

func openAIMessageContentToGeminiParts(content_any any) ([]GeminiPart, error) {
	var parts []GeminiPart

	content, ok := content_any.(string)
	if ok {
		parts = append(parts, GeminiPart{Text: content})
		return parts, nil
	}

	mediaContents, ok := content_any.([]any)
	if !ok {
		return nil, fmt.Errorf("unsupported message content format")
	}

	for _, mediaContent := range mediaContents {
		mediaMap, ok := mediaContent.(map[string]any)
		if !ok {
			continue
		}

		switch mediaMap["type"] {
		case "text":
			parts = append(parts, GeminiPart{Text: mediaMap["text"].(string)})
		case "image_url":
			imageUrl, _ := mediaMap["image_url"].(map[string]any)
			url := imageUrl["url"].(string)
			format, base64, err := service.DecodeBase64FileData(url)
			if err != nil {
				// assume it is a url
				fileData, err := service.GetFileBase64FromUrl(url)
				if err != nil {
					return nil, err
				}
				format = fileData.MimeType
				base64 = fileData.Base64Data
			}
			parts = append(parts, GeminiPart{
				InlineData: &GeminiInlineData{
					MimeType: format,
					Data:     base64,
				},
			})
		}
	}

	return parts, nil
}

func (a *Adaptor) DoRequest(c *gin.Context, info *relaycommon.RelayInfo, requestBody io.Reader) (any, error) {
	return channel.DoApiRequest(a, c, info, requestBody)
}

func (a *Adaptor) DoResponse(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (usage any, err *dto.OpenAIErrorWithStatusCode) {
	if info.RelayMode == constant.RelayModeGemini {
		if info.IsStream {
			return GeminiTextGenerationStreamHandler(c, resp, info)
		} else {
			return GeminiTextGenerationHandler(c, resp, info)
		}
	}

	if strings.HasPrefix(info.UpstreamModelName, "imagen") {
		return GeminiImageHandler(c, resp, info)
	}

	// check if the model is an embedding model
	if strings.HasPrefix(info.UpstreamModelName, "text-embedding") ||
		strings.HasPrefix(info.UpstreamModelName, "embedding") ||
		strings.HasPrefix(info.UpstreamModelName, "gemini-embedding") {
		return GeminiEmbeddingHandler(c, resp, info)
	}

	if info.IsStream {
		err, usage = GeminiChatStreamHandler(c, resp, info)
	} else {
		err, usage = GeminiChatHandler(c, resp, info)
	}

	//if usage.(*dto.Usage).CompletionTokenDetails.ReasoningTokens > 100 {
	//	// 没有请求-thinking的情况下，产生思考token，则按照思考模型计费
	//	if !strings.HasSuffix(info.OriginModelName, "-thinking") &&
	//		!strings.HasSuffix(info.OriginModelName, "-nothinking") {
	//		thinkingModelName := info.OriginModelName + "-thinking"
	//		if operation_setting.SelfUseModeEnabled || helper.ContainPriceOrRatio(thinkingModelName) {
	//			info.OriginModelName = thinkingModelName
	//		}
	//	}
	//}

	return
}

func GeminiImageHandler(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (usage any, err *dto.OpenAIErrorWithStatusCode) {
	responseBody, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, service.OpenAIErrorWrapper(readErr, "read_response_body_failed", http.StatusInternalServerError)
	}
	_ = resp.Body.Close()

	var geminiResponse GeminiImageResponse
	if jsonErr := json.Unmarshal(responseBody, &geminiResponse); jsonErr != nil {
		return nil, service.OpenAIErrorWrapper(jsonErr, "unmarshal_response_body_failed", http.StatusInternalServerError)
	}

	if len(geminiResponse.Predictions) == 0 {
		return nil, service.OpenAIErrorWrapper(errors.New("no images generated"), "no_images", http.StatusBadRequest)
	}

	// convert to openai format response
	openAIResponse := dto.ImageResponse{
		Created: common.GetTimestamp(),
		Data:    make([]dto.ImageData, 0, len(geminiResponse.Predictions)),
	}

	for _, prediction := range geminiResponse.Predictions {
		if prediction.RaiFilteredReason != "" {
			continue // skip filtered image
		}
		openAIResponse.Data = append(openAIResponse.Data, dto.ImageData{
			B64Json: prediction.BytesBase64Encoded,
		})
	}

	jsonResponse, jsonErr := json.Marshal(openAIResponse)
	if jsonErr != nil {
		return nil, service.OpenAIErrorWrapper(jsonErr, "marshal_response_failed", http.StatusInternalServerError)
	}

	c.Writer.Header().Set("Content-Type", "application/json")
	c.Writer.WriteHeader(resp.StatusCode)
	_, _ = c.Writer.Write(jsonResponse)

	// https://github.com/google-gemini/cookbook/blob/719a27d752aac33f39de18a8d3cb42a70874917e/quickstarts/Counting_Tokens.ipynb
	// each image has fixed 258 tokens
	const imageTokens = 258
	generatedImages := len(openAIResponse.Data)

	usage = &dto.Usage{
		PromptTokens:     imageTokens * generatedImages, // each generated image has fixed 258 tokens
		CompletionTokens: 0,                             // image generation does not calculate completion tokens
		TotalTokens:      imageTokens * generatedImages,
	}

	return usage, nil
}

func (a *Adaptor) GetModelList() []string {
	return ModelList
}

func (a *Adaptor) GetChannelName() string {
	return ChannelName
}
