# Oviya Client Applications

Frontend applications that connect to Oviya backend systems.

## 📱 Available Clients

### Mobile App (`mobile/`)
- **Framework**: React Native + Expo
- **Features**:
  - Real-time voice conversations
  - Audio recording/playback
  - Emotion visualization
  - Offline message queuing
- **Platforms**: iOS, Android
- **Connection**: WebSocket to production/services

### Web Client (`web/`)
- **Framework**: Next.js + TypeScript
- **Features**:
  - Browser-based voice interface
  - Real-time audio streaming
  - Responsive design
  - PWA support
- **Connection**: WebSocket + REST API

### Public Website (`website/`)
- **Framework**: Next.js + Tailwind CSS
- **Features**:
  - Marketing pages
  - Demo interface
  - User registration
  - Voice mode integration
- **Connection**: REST API + WebSocket

### Admin Interface (`admin/`)
- **Framework**: React + TypeScript
- **Features**:
  - System monitoring
  - User management
  - Analytics dashboard
  - Configuration management
- **Connection**: REST API to backend

## 🚀 Development Setup

### Prerequisites

```bash
# Node.js 18+
node --version

# Yarn or npm
npm --version

# For mobile development
npm install -g @expo/cli
```

### Client-Specific Setup

#### Mobile App
```bash
cd clients/mobile
npm install
npm start

# iOS simulator
npm run ios

# Android emulator
npm run android
```

#### Web Client
```bash
cd clients/web
npm install
npm run dev
# Access at http://localhost:3000
```

#### Public Website
```bash
cd clients/website
npm install
npm run dev
# Access at http://localhost:3000
```

#### Admin Interface
```bash
cd clients/admin
npm install
npm run dev
# Access at http://localhost:3000
```

## 🔧 Backend Connection

### Environment Configuration

Each client needs backend connection details:

```bash
# .env.local
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/conversation
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_VOICE_MODE=true
```

### WebSocket Communication

```typescript
// clients/web/hooks/useWebSocket.ts
import { useEffect, useRef } from 'react';

export function useOviyaWebSocket(userId: string) {
  const ws = useRef<WebSocket>();

  useEffect(() => {
    ws.current = new WebSocket(
      `${process.env.NEXT_PUBLIC_WS_URL}?user_id=${userId}`
    );

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Handle emotional response
    };

    return () => ws.current?.close();
  }, [userId]);

  const sendAudio = (audioData: ArrayBuffer) => {
    ws.current?.send(JSON.stringify({
      type: 'audio',
      data: arrayBufferToBase64(audioData)
    }));
  };

  return { sendAudio };
}
```

### REST API Integration

```typescript
// clients/web/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL;

export async function getConversationHistory(userId: string) {
  const response = await fetch(`${API_BASE}/conversations?user_id=${userId}`);
  return response.json();
}

export async function sendFeedback(conversationId: string, rating: number) {
  await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    body: JSON.stringify({ conversationId, rating })
  });
}
```

## 🎨 UI/UX Guidelines

### Design System

- **Colors**: Warm, empathetic color palette
- **Typography**: Readable, conversational fonts
- **Animations**: Subtle, emotional transitions
- **Accessibility**: WCAG 2.1 AA compliance

### Voice Interface Patterns

#### Audio Recording States
```typescript
enum RecordingState {
  IDLE = 'idle',
  LISTENING = 'listening',
  PROCESSING = 'processing',
  SPEAKING = 'speaking',
  ERROR = 'error'
}
```

#### Emotion Visualization
```typescript
interface EmotionDisplay {
  emotion: string;
  intensity: number;
  confidence: number;
  color: string;
  icon: string;
}
```

### Error Handling

```typescript
// clients/web/hooks/useErrorHandler.ts
export function useErrorHandler() {
  const handleConnectionError = (error: Error) => {
    // Show user-friendly error message
    toast.error('Connection lost. Reconnecting...');

    // Attempt reconnection
    setTimeout(() => window.location.reload(), 3000);
  };

  const handleAudioError = (error: MediaError) => {
    toast.error('Microphone access needed for voice conversations');
    // Guide user to enable permissions
  };

  return { handleConnectionError, handleAudioError };
}
```

## 📱 Mobile-Specific Features

### Audio Permissions
```typescript
// clients/mobile/hooks/useAudioPermissions.ts
import { Audio } from 'expo-av';

export function useAudioPermissions() {
  const [permission, requestPermission] = Audio.usePermissions();

  useEffect(() => {
    if (permission?.status !== 'granted') {
      requestPermission();
    }
  }, [permission]);

  return permission?.granted ?? false;
}
```

### Background Audio
```typescript
// clients/mobile/services/BackgroundAudio.ts
import * as BackgroundFetch from 'expo-background-fetch';

export class BackgroundAudioService {
  static async registerBackgroundTask() {
    await BackgroundFetch.registerTaskAsync('oviYA-audio-processing', {
      minimumInterval: 15, // minutes
      stopOnTerminate: false,
      startOnBoot: true,
    });
  }
}
```

## 🔒 Security Considerations

### Authentication
- JWT tokens for user sessions
- Secure WebSocket connections (WSS in production)
- API key validation

### Data Privacy
- Local audio processing when possible
- End-to-end encryption for sensitive data
- User consent for data collection

### Content Safety
- Client-side input validation
- Profanity filtering
- Safe conversation boundaries

## 🧪 Testing

### Unit Tests
```bash
# Web client
cd clients/web
npm run test

# Mobile app
cd clients/mobile
npm run test
```

### Integration Tests
```bash
# Test with mock backend
npm run test:e2e

# Test with real backend
BACKEND_URL=http://localhost:8000 npm run test:e2e
```

### Device Testing
```bash
# Mobile device testing
npm run test:devices

# Browser compatibility
npm run test:browsers
```

## 🚀 Deployment

### Web Clients
```bash
# Build for production
npm run build

# Deploy to Vercel/Netlify
npm run deploy
```

### Mobile App
```bash
# Build for stores
npm run build:ios
npm run build:android

# Submit to App Store/Play Store
fastlane beta  # iOS
fastlane beta  # Android
```

### CI/CD
```yaml
# .github/workflows/deploy.yml
name: Deploy Clients
on:
  push:
    branches: [main]
  pull_request:

jobs:
  web:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm run build
      - run: npm run deploy
```

## 📊 Analytics & Monitoring

### Usage Tracking
```typescript
// clients/web/lib/analytics.ts
import { Analytics } from '@segment/analytics-next';

export const analytics = Analytics.load('YOUR_WRITE_KEY');

export function trackConversationStart(userId: string) {
  analytics.track('Conversation Started', {
    userId,
    timestamp: new Date().toISOString(),
    client: 'web'
  });
}

export function trackEmotionDetected(emotion: string, confidence: number) {
  analytics.track('Emotion Detected', {
    emotion,
    confidence,
    client: 'web'
  });
}
```

### Performance Monitoring
```typescript
// clients/web/hooks/usePerformance.ts
import { useEffect } from 'react';

export function usePerformanceMonitoring() {
  useEffect(() => {
    // Monitor WebSocket latency
    const wsLatency = measureWebSocketLatency();

    // Monitor audio processing time
    const audioLatency = measureAudioProcessing();

    // Report to analytics
    reportMetrics({ wsLatency, audioLatency });
  }, []);
}
```

## 🤝 Contributing

### Code Standards
- TypeScript for type safety
- ESLint + Prettier for code quality
- Component library for consistency
- Comprehensive test coverage

### Feature Development
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Create pull request
5. Code review and merge

### Design Reviews
- UX/UI reviews for new features
- Accessibility audits
- Performance impact assessment
- Cross-platform compatibility checks


