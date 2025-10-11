import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
  StatusBar,
  SafeAreaView,
  Dimensions,
} from 'react-native';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';

const { width, height } = Dimensions.get('window');

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState(null);
  const [permissionResponse, requestPermission] = Audio.usePermissions();
  const [isPlaying, setIsPlaying] = useState(false);
  const [sound, setSound] = useState(null);
  const [testResults, setTestResults] = useState([]);
  const [backendStatus, setBackendStatus] = useState('Checking...');
  const [currentEmotion, setCurrentEmotion] = useState('empathetic');
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    checkBackendStatus();
    return sound ? () => sound.unloadAsync() : undefined;
  }, [sound]);

  const checkBackendStatus = async () => {
    try {
      // Test RunPod serverless endpoint directly
      const response = await fetch('https://api.runpod.ai/v2/9sy1v6xggdjiur/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer YOUR_RUNPOD_API_KEY_HERE'
        },
        body: JSON.stringify({
          input: {
            text: 'Test connection',
            emotion: 'empathetic'
          }
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(`✅ RunPod Serverless Connected (ID: ${data.id})`);
        setIsConnected(true);
        addTestResult('success', `RunPod serverless connected: ${data.id}`);
      } else {
        setBackendStatus(`❌ RunPod Error: HTTP ${response.status}`);
        setIsConnected(false);
        addTestResult('error', `RunPod connection failed: HTTP ${response.status}`);
      }
    } catch (err) {
      setBackendStatus('❌ RunPod Offline');
      setIsConnected(false);
      addTestResult('error', `RunPod connection failed: ${err.message}`);
    }
  };

  const addTestResult = (type, message) => {
    setTestResults(prev => [...prev, {
      id: Date.now(),
      type,
      message,
      timestamp: new Date().toLocaleTimeString()
    }]);
  };

  const startRecording = async () => {
    try {
      if (permissionResponse.status !== 'granted') {
        const permission = await requestPermission();
        if (permission.status !== 'granted') {
          Alert.alert('Permission required', 'Microphone permission is required to record audio');
          return;
        }
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(recording);
      setIsRecording(true);
      addTestResult('info', 'Recording started');
    } catch (err) {
      addTestResult('error', `Failed to start recording: ${err.message}`);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;

    try {
      setIsRecording(false);
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      
      addTestResult('success', `Recording completed: ${uri}`);
      
      // Test playing the recorded audio
      await testPlayback(uri);
      
      // Test sending to backend
      await testBackendAudio(uri);
      
      setRecording(null);
    } catch (err) {
      addTestResult('error', `Failed to stop recording: ${err.message}`);
    }
  };

  const testPlayback = async (uri) => {
    try {
      const { sound } = await Audio.Sound.createAsync({ uri });
      setSound(sound);
      
      await sound.playAsync();
      setIsPlaying(true);
      
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.didJustFinish) {
          setIsPlaying(false);
          addTestResult('success', 'Audio playback completed');
        }
      });
      
      addTestResult('success', 'Audio playback started');
    } catch (err) {
      addTestResult('error', `Playback failed: ${err.message}`);
    }
  };

  const testBackendAudio = async (uri) => {
    try {
      // Read audio file
      const fileInfo = await FileSystem.getInfoAsync(uri);
      addTestResult('info', `Audio file size: ${Math.round(fileInfo.size / 1024)}KB`);
      
      // Test ASR service
      const audioData = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      const response = await fetch('http://192.168.1.12:8001/transcribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio: audioData,
          sample_rate: 16000
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        addTestResult('success', `ASR Result: "${result.text}"`);
        
        // Test CSM service with transcribed text
        await testCSMService(result.text);
      } else {
        addTestResult('error', `ASR failed: HTTP ${response.status}`);
      }
    } catch (err) {
      addTestResult('error', `Backend audio test failed: ${err.message}`);
    }
  };

  const testCSMService = async (text) => {
    try {
      // Use RunPod serverless endpoint directly
      const response = await fetch('https://api.runpod.ai/v2/9sy1v6xggdjiur/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer YOUR_RUNPOD_API_KEY_HERE'
        },
        body: JSON.stringify({
          input: {
            text: text || 'Hello, this is a test of the CSM service',
            emotion: currentEmotion
          }
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        addTestResult('success', `RunPod CSM Request ID: ${result.id}`);
        
        // Check status after a short delay
        setTimeout(async () => {
          try {
            const statusResponse = await fetch(`https://api.runpod.ai/v2/9sy1v6xggdjiur/status/${result.id}`, {
              headers: {
                'Authorization': 'Bearer YOUR_RUNPOD_API_KEY_HERE'
              }
            });
            
            if (statusResponse.ok) {
              const statusResult = await statusResponse.json();
              if (statusResult.status === 'COMPLETED') {
                addTestResult('success', `🎵 Audio generated! Duration: ${statusResult.output.duration}s`);
                addTestResult('success', `Text: "${statusResult.output.text}"`);
              } else if (statusResult.status === 'FAILED') {
                addTestResult('error', `RunPod failed: ${statusResult.error}`);
              } else {
                addTestResult('info', `RunPod status: ${statusResult.status}`);
              }
            }
          } catch (err) {
            addTestResult('error', `Status check failed: ${err.message}`);
          }
        }, 2000);
        
      } else {
        addTestResult('error', `RunPod CSM failed: HTTP ${response.status}`);
      }
    } catch (err) {
      addTestResult('error', `RunPod CSM test failed: ${err.message}`);
    }
  };

  const testCompletePipeline = async () => {
    try {
      // Create session
      const sessionResponse = await fetch('http://192.168.1.12:8002/session/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: 'test_user' })
      });
      
      if (sessionResponse.ok) {
        const sessionData = await sessionResponse.json();
        addTestResult('success', `Session created: ${sessionData.session_id}`);
        
        // Send test message
        const messageResponse = await fetch(`http://192.168.1.12:8002/session/${sessionData.session_id}/message`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: 'Hello, this is a test of the complete Oviya AI pipeline!',
            emotion: currentEmotion,
            priority: 'normal'
          })
        });
        
        if (messageResponse.ok) {
          const result = await messageResponse.json();
          addTestResult('success', `Complete pipeline test: "${result.text}"`);
        } else {
          addTestResult('error', `Pipeline test failed: HTTP ${messageResponse.status}`);
        }
      } else {
        addTestResult('error', `Session creation failed: HTTP ${sessionResponse.status}`);
      }
    } catch (error) {
      addTestResult('error', `Pipeline test failed: ${error.message}`);
    }
  };

  const testBackend = async () => {
    await checkBackendStatus();
  };

  const refreshStatus = async () => {
    await checkBackendStatus();
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      empathetic: '#4CAF50',
      encouraging: '#FF9800',
      calm: '#2196F3',
      joyful: '#FFC107',
      concerned: '#FF5722'
    };
    return colors[emotion] || '#4CAF50';
  };

  const getEmotionIcon = (emotion) => {
    const icons = {
      empathetic: '💚',
      encouraging: '🧡',
      calm: '💙',
      joyful: '💛',
      concerned: '❤️'
    };
    return icons[emotion] || '💚';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a1a" />
      
      <View style={styles.header}>
        <Text style={styles.title}>🎤 Oviya AI</Text>
        <Text style={styles.subtitle}>Real-time Voice AI Test</Text>
      </View>
      
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>{backendStatus}</Text>
        <Text style={styles.statusText}>
          Status: {isRecording ? '🔴 Recording...' : isPlaying ? '🔊 Playing...' : '✅ Ready'}
        </Text>
        <Text style={styles.statusText}>
          Permission: {permissionResponse?.status || 'Unknown'}
        </Text>
        <Text style={styles.statusText}>
          Emotion: {getEmotionIcon(currentEmotion)} {currentEmotion}
        </Text>
      </View>
      
      {/* Emotion Selector */}
      <View style={styles.emotionContainer}>
        <Text style={styles.emotionTitle}>Choose Emotion:</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.emotionScroll}>
          {['empathetic', 'encouraging', 'calm', 'joyful', 'concerned'].map((emotion) => (
            <TouchableOpacity
              key={emotion}
              style={[
                styles.emotionButton,
                { backgroundColor: getEmotionColor(emotion) },
                currentEmotion === emotion && styles.emotionButtonActive
              ]}
              onPress={() => setCurrentEmotion(emotion)}
            >
              <Text style={styles.emotionText}>
                {getEmotionIcon(emotion)} {emotion}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>
      
      {/* Main Recording Button */}
      <View style={styles.recordingContainer}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording && styles.recordButtonActive,
            { backgroundColor: getEmotionColor(currentEmotion) }
          ]}
          onPress={isRecording ? stopRecording : startRecording}
        >
          <Text style={styles.recordButtonText}>
            {isRecording ? '🛑 Stop Recording' : '🎤 Start Recording'}
          </Text>
        </TouchableOpacity>
      </View>
      
      {/* Control Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, styles.testButton]}
          onPress={testCompletePipeline}
        >
          <Text style={styles.buttonText}>🧠 Test Complete Pipeline</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.button, styles.refreshButton]}
          onPress={refreshStatus}
        >
          <Text style={styles.buttonText}>🔄 Refresh Status</Text>
        </TouchableOpacity>
      </View>
      
      {/* Results */}
      <ScrollView style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>📋 Test Results:</Text>
        {testResults.map((result) => (
          <View key={result.id} style={[styles.resultItem, styles[result.type]]}>
            <Text style={styles.resultText}>
              [{result.timestamp}] {result.message}
            </Text>
          </View>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#ccc',
  },
  statusContainer: {
    backgroundColor: '#2a2a2a',
    padding: 15,
    borderRadius: 15,
    marginBottom: 20,
  },
  statusText: {
    fontSize: 14,
    color: '#fff',
    marginBottom: 5,
  },
  emotionContainer: {
    marginBottom: 20,
  },
  emotionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  emotionScroll: {
    flexDirection: 'row',
  },
  emotionButton: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 10,
    opacity: 0.7,
  },
  emotionButtonActive: {
    opacity: 1,
    transform: [{ scale: 1.1 }],
  },
  emotionText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  recordingContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  recordButton: {
    width: 200,
    height: 200,
    borderRadius: 100,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  recordButtonActive: {
    transform: [{ scale: 1.1 }],
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
    flex: 1,
    marginHorizontal: 5,
  },
  testButton: {
    backgroundColor: '#4CAF50',
  },
  refreshButton: {
    backgroundColor: '#FF9800',
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: '#2a2a2a',
    borderRadius: 15,
    padding: 15,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  resultItem: {
    padding: 10,
    marginBottom: 5,
    borderRadius: 8,
  },
  success: {
    backgroundColor: 'rgba(76, 175, 80, 0.2)',
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  error: {
    backgroundColor: 'rgba(244, 67, 54, 0.2)',
    borderLeftWidth: 4,
    borderLeftColor: '#F44336',
  },
  info: {
    backgroundColor: 'rgba(33, 150, 243, 0.2)',
    borderLeftWidth: 4,
    borderLeftColor: '#2196F3',
  },
  resultText: {
    fontSize: 12,
    color: '#fff',
  },
});