# Oviya Web Frontend

Real-time voice AI web application built with Next.js 14, TypeScript, and Tailwind CSS.

## Features

- 🎤 Real-time voice recording and playback
- 🤖 AI conversation with emotional responses
- 🎨 Beautiful, responsive UI with Framer Motion animations
- 🔌 WebSocket integration for real-time communication
- 🎵 Multiple emotion modes (empathetic, encouraging, calm, joyful, concerned)
- 📱 Mobile-friendly design
- 🔒 Privacy-first approach

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS, Framer Motion
- **Audio**: Web Audio API, MediaRecorder API
- **Communication**: Socket.io-client
- **State Management**: React Hooks, Zustand
- **Notifications**: React Hot Toast

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Running Oviya Orchestrator service

### Installation

1. Install dependencies:
```bash
npm install
```

2. Set environment variables:
```bash
cp .env.example .env.local
```

Edit `.env.local`:
```
NEXT_PUBLIC_ORCHESTRATOR_URL=http://localhost:8002
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
oviya-web/
├── components/          # React components
│   ├── VoiceChat.tsx    # Main voice chat interface
│   ├── EmotionSelector.tsx
│   ├── ConversationHistory.tsx
│   ├── ConnectionStatus.tsx
│   ├── Header.tsx
│   └── Footer.tsx
├── hooks/               # Custom React hooks
│   ├── useVoiceChat.ts
│   ├── useAudioRecorder.ts
│   └── useAudioPlayer.ts
├── pages/               # Next.js pages
│   └── index.tsx        # Home page
├── styles/              # Global styles
├── public/              # Static assets
└── types/               # TypeScript type definitions
```

## Usage

1. **Connect**: The app automatically connects to the Orchestrator service
2. **Choose Emotion**: Select Oviya's emotional tone
3. **Start Talking**: Click the microphone button to start recording
4. **Listen**: Oviya will respond with voice and text
5. **Interrupt**: Click the microphone while Oviya is speaking to interrupt

## API Integration

The web app communicates with the Oviya Orchestrator service via WebSocket:

- **Connection**: `ws://localhost:8002/session/{session_id}/stream`
- **Message Types**: `message`, `interrupt`, `response`, `interrupt_result`
- **Audio Format**: WebM/Opus, 16kHz, mono

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

### Code Style

- ESLint + Prettier for code formatting
- TypeScript for type safety
- Tailwind CSS for styling
- Framer Motion for animations

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Connect repository to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy automatically

### Other Platforms

Build the app:
```bash
npm run build
```

The built app will be in the `.next` directory.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_ORCHESTRATOR_URL` | Orchestrator service URL | `http://localhost:8002` |

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: [Report bugs and request features](https://github.com/oviya-ai/oviya-web/issues)
- Email: hello@oviya.ai
- Twitter: [@oviya_ai](https://twitter.com/oviya_ai)


