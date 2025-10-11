# ASR Serverless Endpoint

[![Runpod](https://api.runpod.io/badge/jarvis138/asr-serverless)](https://console.runpod.io/hub/jarvis138/asr-serverless)

## Silero VAD + Whisper ASR Pipeline

This RunPod serverless endpoint provides:

- **Silero VAD**: Voice Activity Detection with <1ms latency
- **Whisper ASR**: Speech recognition with GPU acceleration
- **Real-time Processing**: Optimized for streaming audio

## API Usage

### VAD Only
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "audio": "base64_encoded_audio",
      "sample_rate": 16000,
      "operation": "vad"
    }
  }'
```

### Transcription Only
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "audio": "base64_encoded_audio",
      "sample_rate": 16000,
      "operation": "transcribe"
    }
  }'
```

### Both VAD + Transcription
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "audio": "base64_encoded_audio",
      "sample_rate": 16000,
      "operation": "both"
    }
  }'
```

## Response Format

```json
{
  "text": "transcribed text",
  "confidence": 0.95,
  "is_speech": true,
  "speech_prob": 0.98,
  "segments": [...],
  "language": "en",
  "timestamp": 1234567890.123
}
```


