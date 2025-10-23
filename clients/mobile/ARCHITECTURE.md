## oviya-mobile — Architecture

### Purpose
React Native app for mobile voice testing.

### Components
- `App.tsx`: navigation and root providers
- `AudioTestScreen.tsx`: mic capture, send to backend, playback

### Data Flow
- RN mic → PCM via native bridge → backend WS → streaming audio back → RN playback

### Notes
- Setup scripts, TypeScript config; assets and icons for app stores.




