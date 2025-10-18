## oviya-website — Architecture

### Purpose
Marketing/demo site (Next.js) with live voice mode demo components and docs.

### Components
- Demo widgets: `LiveAIDemo.tsx`, `VoiceMode.tsx`, `ChatGPTVoiceMode.tsx`
- Hooks: `useLiveDemo`, `useVoiceMode`
- Pages: `_app.tsx`, `index.tsx`, `demo.tsx`

### Data Flow
- Browser mic → demo hook → backend WS/RTC → receive chunks → WebAudio playback

### Notes
- Dockerized for deployment; Tailwind; start script `start-voice-mode.sh`.


