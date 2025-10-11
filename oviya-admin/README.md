# Oviya Admin Dashboard

Admin dashboard for monitoring and managing the Oviya AI system.

## Features

- 📊 **Real-time Metrics**: Monitor system performance, latency, and throughput
- 👥 **User Management**: View and manage user accounts and sessions
- 🔧 **Service Monitoring**: Track health status of all services
- 📈 **Analytics**: Detailed analytics and usage statistics
- 🛡️ **Security**: Monitor security events and access logs
- ⚙️ **System Settings**: Configure system parameters and features

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS, Framer Motion
- **Charts**: Recharts for data visualization
- **State Management**: Zustand
- **Forms**: React Hook Form with Zod validation
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Access to Oviya AI services

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
NEXT_PUBLIC_API_URL=http://localhost:8002
NEXT_PUBLIC_WS_URL=ws://localhost:8002
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
oviya-admin/
├── components/          # React components
│   ├── DashboardLayout.tsx
│   ├── OverviewCards.tsx
│   ├── SystemMetrics.tsx
│   ├── ServiceStatus.tsx
│   └── RecentActivity.tsx
├── pages/               # Next.js pages
│   ├── index.tsx        # Dashboard
│   ├── users.tsx        # User management
│   ├── analytics.tsx    # Analytics
│   ├── services.tsx     # Service monitoring
│   ├── monitoring.tsx   # System monitoring
│   ├── security.tsx     # Security logs
│   └── settings.tsx     # System settings
├── hooks/               # Custom React hooks
├── utils/               # Utility functions
├── types/               # TypeScript type definitions
└── styles/              # Global styles
```

## Features Overview

### Dashboard
- System overview with key metrics
- Real-time performance charts
- Service health status
- Recent activity feed

### User Management
- User account overview
- Session management
- Usage statistics
- Account actions

### Analytics
- Usage patterns and trends
- Performance metrics
- Error analysis
- Custom date ranges

### Service Monitoring
- Real-time service health
- Resource utilization
- Performance metrics
- Alert management

### Security
- Security event logs
- Access monitoring
- Threat detection
- Compliance reporting

### Settings
- System configuration
- Feature toggles
- User permissions
- Integration settings

## API Integration

The admin dashboard connects to the Oviya AI services:

- **Orchestrator Service**: Main API for system data
- **WebSocket**: Real-time updates and monitoring
- **Authentication**: Secure admin access
- **Data Export**: Export logs and reports

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
- Component-based architecture

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Connect repository to Vercel
3. Set environment variables
4. Deploy automatically

### Other Platforms

Build the app:
```bash
npm run build
```

The built app will be in the `.next` directory.

## Security

- Admin authentication required
- Role-based access control
- Secure API communication
- Audit logging
- Data encryption

## Monitoring

- Real-time system metrics
- Performance monitoring
- Error tracking
- User activity logs
- Security event monitoring

## Support

- GitHub Issues: [Report bugs and request features](https://github.com/oviya-ai/oviya-admin/issues)
- Email: admin@oviya.ai
- Documentation: [Admin Guide](https://docs.oviya.ai/admin)

## License

MIT License - see LICENSE file for details.


