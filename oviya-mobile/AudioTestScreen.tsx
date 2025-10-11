import React, { useState, useEffect } from 'react'
import { View, Text, TouchableOpacity, StyleSheet, Alert, PermissionsAndroid, Platform } from 'react-native'
import { Audio } from 'expo-av'
import * as FileSystem from 'expo-file-system'

export default function AudioTestScreen() {
  const [isRecording, setIsRecording] = useState(false)
  const [recording, setRecording] = useState(null)
  const [permissionResponse, requestPermission] = Audio.usePermissions()
  const [isPlaying, setIsPlaying] = useState(false)
  const [sound, setSound] = useState(null)
  const [testResults, setTestResults] = useState([])

  useEffect(() => {
    return sound
      ? () => {
          sound.unloadAsync()
        }
      : undefined
  }, [sound])

  const startRecording = async () => {
    try {
      if (permissionResponse.status !== 'granted') {
        const permission = await requestPermission()
        if (permission.status !== 'granted') {
          Alert.alert('Permission required', 'Microphone permission is required to record audio')
          return
        }
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      })

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      )
      setRecording(recording)
      setIsRecording(true)
      console.log('Recording started')
    } catch (err) {
      console.error('Failed to start recording', err)
      Alert.alert('Error', 'Failed to start recording')
    }
  }

  const stopRecording = async () => {
    if (!recording) return

    try {
      setIsRecording(false)
      await recording.stopAndUnloadAsync()
      const uri = recording.getURI()
      console.log('Recording stopped and stored at', uri)
      
      // Test the audio file
      await testAudioFile(uri)
      
      setRecording(null)
    } catch (err) {
      console.error('Failed to stop recording', err)
      Alert.alert('Error', 'Failed to stop recording')
    }
  }

  const testAudioFile = async (uri) => {
    try {
      // Read audio file info
      const fileInfo = await FileSystem.getInfoAsync(uri)
      console.log('Audio file info:', fileInfo)
      
      // Test playing the recorded audio
      const { sound } = await Audio.Sound.createAsync({ uri })
      setSound(sound)
      
      await sound.playAsync()
      setIsPlaying(true)
      
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.didJustFinish) {
          setIsPlaying(false)
        }
      })
      
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'success',
        message: `Audio recorded successfully (${Math.round(fileInfo.size / 1024)}KB)`,
        timestamp: new Date().toLocaleTimeString()
      }])
      
    } catch (err) {
      console.error('Failed to test audio file', err)
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        message: `Failed to test audio: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }])
    }
  }

  const testBackendConnection = async () => {
    try {
      const response = await fetch('http://localhost:8002/health')
      const data = await response.json()
      
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'success',
        message: `Backend connected: ${data.status}`,
        timestamp: new Date().toLocaleTimeString()
      }])
    } catch (err) {
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        message: `Backend connection failed: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }])
    }
  }

  const testASRService = async () => {
    try {
      // This would normally send audio to ASR service
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'info',
        message: 'ASR service test - would send audio for transcription',
        timestamp: new Date().toLocaleTimeString()
      }])
    } catch (err) {
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        message: `ASR test failed: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }])
    }
  }

  const testCSMService = async () => {
    try {
      const response = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: 'Hello, this is a test of the CSM service',
          emotion: 'empathetic',
          session_id: 'test_session'
        })
      })
      
      const data = await response.json()
      
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'success',
        message: `CSM service responded: ${data.text}`,
        timestamp: new Date().toLocaleTimeString()
      }])
    } catch (err) {
      setTestResults(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        message: `CSM test failed: ${err.message}`,
        timestamp: new Date().toLocaleTimeString()
      }])
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Oviya AI Audio Test</Text>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, isRecording ? styles.recordingButton : styles.recordButton]}
          onPress={isRecording ? stopRecording : startRecording}
        >
          <Text style={styles.buttonText}>
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.button, styles.testButton]}
          onPress={testBackendConnection}
        >
          <Text style={styles.buttonText}>Test Backend</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.button, styles.testButton]}
          onPress={testASRService}
        >
          <Text style={styles.buttonText}>Test ASR</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.button, styles.testButton]}
          onPress={testCSMService}
        >
          <Text style={styles.buttonText}>Test CSM</Text>
        </TouchableOpacity>
      </View>
      
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          Status: {isRecording ? 'Recording...' : isPlaying ? 'Playing...' : 'Ready'}
        </Text>
        <Text style={styles.statusText}>
          Permission: {permissionResponse?.status || 'Unknown'}
        </Text>
      </View>
      
      <View style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>Test Results:</Text>
        {testResults.map((result) => (
          <View key={result.id} style={[styles.resultItem, styles[result.type]]}>
            <Text style={styles.resultText}>
              [{result.timestamp}] {result.message}
            </Text>
          </View>
        ))}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
    color: '#333',
  },
  buttonContainer: {
    marginBottom: 30,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: '#007AFF',
  },
  recordingButton: {
    backgroundColor: '#FF3B30',
  },
  testButton: {
    backgroundColor: '#34C759',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  statusContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  statusText: {
    fontSize: 14,
    marginBottom: 5,
    color: '#666',
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  resultItem: {
    padding: 8,
    marginBottom: 5,
    borderRadius: 5,
  },
  success: {
    backgroundColor: '#E8F5E8',
  },
  error: {
    backgroundColor: '#FFE8E8',
  },
  info: {
    backgroundColor: '#E8F0FF',
  },
  resultText: {
    fontSize: 12,
    color: '#333',
  },
})


