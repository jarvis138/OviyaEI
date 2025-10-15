# Oviya AI Website

A clean, minimalist website for Oviya AI that showcases real-time voice interaction capabilities.

## Features

- **Clean Design**: Inspired by Sesame's minimalism with Oviya.site's aesthetic
- **Live AI Demo**: Real-time voice conversation without signup
- **WebSocket Integration**: Connects to your existing oviya-production backend
- **Responsive**: Mobile-first design that works on all devices
- **Modern Stack**: Next.js 14, TypeScript, Tailwind CSS, Framer Motion

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local with your WebSocket server URL
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   ```
   http://localhost:3000
   ```

## Backend Integration

This website connects to your existing Oviya AI backend:

- **WebSocket Server**: `http://localhost:8000` (your oviya-production)
- **Real-time Audio**: Streams audio to/from your voice processing pipeline
- **Emotion Detection**: Displays detected emotions from your AI system
- **Conversation History**: Shows recent messages and responses

## Project Structure

```
oviya-website/
├── components/           # React components
│   ├── HeroSection.tsx   # Landing hero
│   ├── LiveAIDemo.tsx    # Interactive demo
│   ├── FeatureShowcase.tsx # Capabilities
│   ├── SimpleNav.tsx    # Navigation
│   └── CleanFooter.tsx   # Footer
├── hooks/               # Custom hooks
│   └── useLiveDemo.ts   # WebSocket integration
├── pages/               # Next.js pages
│   └── index.tsx        # Main page
├── styles/              # Global styles
│   └── globals.css      # Tailwind + custom CSS
└── public/              # Static assets
```

## Design System

### Colors
- **Primary**: Purple (#8b5cf6) and Blue (#3b82f6)
- **Background**: Soft gradients from purple-50 to blue-50
- **Text**: Gray-900 for headings, Gray-600 for body

### Typography
- **Font**: Inter (clean, modern sans-serif)
- **Sizes**: Responsive scaling from mobile to desktop

### Components
- **Clean minimalism**: Lots of white space, simple layouts
- **Smooth animations**: Framer Motion for interactions
- **Glass effects**: Backdrop blur for modern feel

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript checks

### Code Style

- **TypeScript**: Strict mode enabled
- **ESLint**: Next.js recommended rules
- **Prettier**: Automatic code formatting
- **Tailwind**: Utility-first CSS

## Deployment

### Vercel (Recommended)
```bash
npm run build
# Deploy to Vercel
```

### Docker
```bash
docker build -t oviya-website .
docker run -p 3000:3000 oviya-website
```

## Backend Requirements

Make sure your oviya-production backend is running:

```bash
cd oviya-production
python websocket_server.py
```

The website will connect to `http://localhost:8000` by default.

## Customization

### Environment Variables
- `NEXT_PUBLIC_ORCHESTRATOR_URL`: WebSocket server URL
- `NEXT_PUBLIC_ANALYTICS_ID`: Analytics tracking ID
- `NEXT_PUBLIC_SENTRY_DSN`: Error monitoring

### Styling
- Edit `tailwind.config.js` for theme customization
- Modify `styles/globals.css` for custom CSS
- Update components for layout changes

## Support

For issues or questions:
- GitHub: [oviya-ai/oviya-website](https://github.com/oviya-ai/oviya-website)
- Email: hello@oviya.ai

---

Built with ❤️ for Oviya AI
