# 🎤 Oviya AI Mobile App

A React Native mobile app for testing the Oviya AI voice system with real-time audio recording, playback, and backend integration.

## 🚀 Features

- **Real-time Audio Recording**: High-quality audio capture with visual feedback
- **Audio Playback**: Test recorded audio with built-in player
- **Backend Integration**: Connect to ASR, CSM, and Orchestrator services
- **Emotion Selection**: Choose from 5 different emotional tones
- **Complete Pipeline Testing**: Test the full ASR → GPT → CSM workflow
- **Live Status Monitoring**: Real-time backend service status
- **Test Results Logging**: Detailed logs of all test operations

## 📱 Setup

### Prerequisites
- Node.js installed
- Expo CLI (`npm install -g expo-cli`)
- Expo Go app on your phone

### Installation
```bash
cd oviya-mobile
npm install
```

### Start Development Server
```bash
npm start
```

### Connect Your Phone
1. Install **Expo Go** from App Store (iOS) or Google Play (Android)
2. Make sure your phone and computer are on the same WiFi network
3. Open Expo Go and scan the QR code or enter URL manually:
   ```
   exp://192.168.1.12:8083
   ```

## 🎯 Testing Features

### 1. Audio Recording
- Tap the large record button to start/stop recording
- Visual feedback shows recording status
- Automatic permission handling

### 2. Emotion Selection
- Choose from 5 emotions: empathetic, encouraging, calm, joyful, concerned
- Each emotion has a unique color and icon
- Affects the AI's response tone

### 3. Backend Testing
- **Complete Pipeline Test**: Tests ASR → GPT → CSM workflow
- **Status Refresh**: Checks backend service health
- **Real-time Logging**: See all test results in the log

### 4. Service Integration
- **ASR Service**: Speech-to-text conversion
- **CSM Service**: Text-to-speech with emotions
- **Orchestrator**: Coordinates the complete pipeline

## 🔧 Backend Services

The app connects to these services running on your computer:

- **CSM Service**: `http://192.168.1.12:8000`
- **ASR Service**: `http://192.168.1.12:8001`
- **Orchestrator**: `http://192.168.1.12:8002`

## 📊 Test Results

All test operations are logged with:
- Timestamp
- Operation type (success/error/info)
- Detailed messages
- Color-coded results

## 🎨 UI Features

- **Dark Theme**: Modern dark interface
- **Responsive Design**: Works on all screen sizes
- **Visual Feedback**: Color-coded status indicators
- **Smooth Animations**: Button press effects and transitions

## 🚨 Troubleshooting

### Connection Issues
- Ensure all backend services are running
- Check WiFi network connectivity
- Verify IP address in the app matches your computer

### Audio Issues
- Grant microphone permissions when prompted
- Check device audio settings
- Ensure no other apps are using the microphone

### Backend Errors
- Check service logs for detailed error messages
- Verify all services are healthy
- Restart services if needed

## 📱 Supported Platforms

- **iOS**: iPhone and iPad
- **Android**: All Android devices
- **Web**: Browser testing (limited audio features)

## 🔄 Development

### File Structure
```
oviya-mobile/
├── App.tsx          # Main application component
├── index.js         # Entry point
├── app.json         # Expo configuration
├── package.json     # Dependencies
└── babel.config.js  # Babel configuration
```

### Key Dependencies
- `expo-av`: Audio recording and playback
- `expo-file-system`: File operations
- `socket.io-client`: WebSocket communication
- `@react-native-async-storage/async-storage`: Local storage

## 🎉 Ready to Test!

The app is now ready for comprehensive audio testing. Connect your phone and start exploring the Oviya AI voice system!