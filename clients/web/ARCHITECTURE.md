## oviya-web — Architecture

### Purpose
Web client for realtime voice chat (Next.js + React), with audio capture, playback, and conversation UI.

### Components
- `components/VoiceChat.tsx`: main chat widget
- `components/ConnectionStatus.tsx`, `ConversationHistory.tsx`, `EmotionSelector.tsx`, layout components
- Hooks: `useAudioRecorder`, `useAudioPlayer`, `useVoiceChat`

### Data Flow
- Recorder (MediaStream) → PCM chunks → WebSocket/WebRTC backend
- Receive audio chunks → buffered playback with `useAudioPlayer` (jitter-aware)
- UI updates conversation history and status

### Notes
- Tailwind setup, TypeScript config, Next.js pages under `pages/`




