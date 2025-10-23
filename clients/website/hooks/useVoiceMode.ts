/**
 * Production Voice Mode Hook - Fixed and tested
 * Implements continuous audio streaming with proper error handling
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { toast } from 'react-hot-toast'

// Constants
const SAMPLE_RATE = 16000
const CHUNK_DURATION_MS = 250
const SILENCE_THRESHOLD_MS = 1500
const MAX_RECORDING_MS = 60000  // 60 seconds for longer conversations

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  emotion?: string
  timestamp: number
}

interface VoiceModeState {
  isConnected: boolean
  isListening: boolean
  isSpeaking: boolean
  currentEmotion: string | null
  messages: Message[]
  audioLevel: number
  error: string | null
  toggleVoiceMode: () => void
}

export const useVoiceMode = (): VoiceModeState => {
  const [isConnected, setIsConnected] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [audioLevel, setAudioLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [hasStarted, setHasStarted] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const playbackContextRef = useRef<AudioContext | null>(null)
  const isPlayingRef = useRef(false)
  const continuousModeRef = useRef(true)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Connect to WebSocket
  useEffect(() => {
    const connectWebSocket = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) return
      
      try {
        console.log('🔄 Connecting to WebSocket...')
        const ws = new WebSocket('ws://localhost:8000/ws/conversation?user_id=web_user')
        
        ws.onopen = () => {
          console.log('✅ WebSocket connected')
          setIsConnected(true)
          setError(null)
          wsRef.current = ws
          toast.success('Connected to Oviya')
        }

        ws.onmessage = async (event) => {
          try {
            const data = JSON.parse(event.data)
            console.log('📨 Message:', data.type)
            
            switch (data.type) {
              case 'transcription':
                const userMessage: Message = {
                  id: `user-${Date.now()}`,
                  role: 'user',
                  text: data.text,
                  timestamp: Date.now()
                }
                setMessages(prev => [...prev, userMessage])
                break
                
              case 'response':
                console.log('📨 Response received:', {
                  text: data.text,
                  emotion: data.emotion,
                  audio_chunks: data.audio_chunks?.length || 0,
                  duration: data.duration
                })
                
                const assistantMessage: Message = {
                  id: `assistant-${Date.now()}`,
                  role: 'assistant',
                  text: data.text,
                  emotion: data.emotion,
                  timestamp: Date.now()
                }
                setMessages(prev => [...prev, assistantMessage])
                setCurrentEmotion(data.emotion)
                
                // Play audio if available
                if (data.audio_chunks && data.audio_chunks.length > 0) {
                  console.log(`🎵 Received ${data.audio_chunks.length} audio chunks, first chunk: ${data.audio_chunks[0].substring(0, 50)}...`)
                  await playAudioChunks(data.audio_chunks)
                } else {
                  console.warn('⚠️ No audio chunks in response!')
                }
                break
                
              case 'error':
                console.error('Server error:', data.message)
                toast.error(data.message)
                break
            }
          } catch (e) {
            console.error('Error parsing message:', e)
          }
        }

        ws.onerror = (event) => {
          console.error('WebSocket error:', event)
          setError('Connection error')
        }

        ws.onclose = () => {
          console.log('❌ WebSocket disconnected')
          setIsConnected(false)
          wsRef.current = null
          
          // Reconnect after 2 seconds
          if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000)
        }
      } catch (err) {
        console.error('Failed to connect:', err)
        setError('Failed to connect')
        setTimeout(connectWebSocket, 2000)
      }
    }

    connectWebSocket()

    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  // Play audio chunks with precise timing for gapless playback
  const playAudioChunks = async (chunks: string[]) => {
    try {
      console.log(`🎵 Playing ${chunks.length} audio chunks...`)
      setIsSpeaking(true)
      
      // Get or create playback context
      if (!playbackContextRef.current) {
        playbackContextRef.current = new AudioContext({ sampleRate: 24000 })
      }
      
      const audioContext = playbackContextRef.current
      
      // Resume if suspended
      if (audioContext.state === 'suspended') {
        await audioContext.resume()
        console.log('🔊 Audio context resumed')
      }
      
      // Decode all chunks first
      const audioBuffers: AudioBuffer[] = []
      for (const chunk of chunks) {
        try {
          // Decode base64
          const binaryString = atob(chunk)
          const bytes = new Uint8Array(binaryString.length)
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i)
          }
          
          // Convert to float32
          const int16 = new Int16Array(bytes.buffer)
          const float32 = new Float32Array(int16.length)
          for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0
          }
          
          // Create audio buffer
          const audioBuffer = audioContext.createBuffer(1, float32.length, 24000)
          audioBuffer.getChannelData(0).set(float32)
          audioBuffers.push(audioBuffer)
        } catch (e) {
          console.error('Error decoding chunk:', e)
        }
      }
      
      console.log(`✅ Decoded ${audioBuffers.length} audio buffers`)
      
      // Play all buffers with precise timing to avoid gaps
      let startTime = audioContext.currentTime
      const sources: AudioBufferSourceNode[] = []
      
      audioBuffers.forEach((buffer) => {
        const source = audioContext.createBufferSource()
        source.buffer = buffer
        source.connect(audioContext.destination)
        source.start(startTime)
        sources.push(source)
        startTime += buffer.duration
      })
      
      // Calculate total duration
      const totalDuration = audioBuffers.reduce((sum, buf) => sum + buf.duration, 0)
      console.log(`⏱️  Total audio duration: ${totalDuration.toFixed(2)}s`)
      
      // Wait for all audio to finish
      await new Promise(resolve => setTimeout(resolve, totalDuration * 1000))
      
      console.log('✅ Audio playback complete')
      setIsSpeaking(false)
      
      // Auto-resume listening in continuous mode
      if (continuousModeRef.current && isConnected) {
        console.log('🔄 Auto-resuming listening...')
        setTimeout(() => startListening(), 500)
      }
    } catch (error) {
      console.error('Error playing audio:', error)
      setIsSpeaking(false)
    }
  }

  // Start listening
  const startListening = async () => {
    if (!isConnected || isListening || isSpeaking) {
      console.log('Cannot start listening:', { isConnected, isListening, isSpeaking })
      return
    }
    
    try {
      console.log('🎤 Starting microphone...')
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })
      
      streamRef.current = stream
      
      // Create audio context at device sample rate
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor
      
      let silenceStart: number | null = null
      let recordingTimeout: NodeJS.Timeout | null = null
      
      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
        
        const inputData = e.inputBuffer.getChannelData(0)
        
        // Calculate audio level
        let sum = 0
        for (let i = 0; i < inputData.length; i++) {
          sum += Math.abs(inputData[i])
        }
        const level = sum / inputData.length
        setAudioLevel(level * 10) // Scale for visibility
        
        // Detect silence for VAD (more lenient threshold)
        const SILENCE_THRESHOLD = 0.005  // Lower threshold, less sensitive
        const now = Date.now()
        
        if (level < SILENCE_THRESHOLD) {
          if (!silenceStart) {
            silenceStart = now
          } else if (now - silenceStart > SILENCE_THRESHOLD_MS + 500) {
            // Add extra 500ms buffer before stopping
            console.log('🔇 Silence detected, stopping...')
            stopListening()
            return
          }
        } else {
          silenceStart = null
        }
        
        // Downsample to 16kHz if needed
        const inputSampleRate = audioContext.sampleRate
        let pcmData: Int16Array
        
        if (inputSampleRate !== SAMPLE_RATE) {
          // Simple downsampling
          const ratio = inputSampleRate / SAMPLE_RATE
          const newLength = Math.floor(inputData.length / ratio)
          pcmData = new Int16Array(newLength)
          
          for (let i = 0; i < newLength; i++) {
            const srcIndex = Math.floor(i * ratio)
            const sample = Math.max(-1, Math.min(1, inputData[srcIndex]))
            pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
          }
        } else {
          // Direct conversion
          pcmData = new Int16Array(inputData.length)
          for (let i = 0; i < inputData.length; i++) {
            const sample = Math.max(-1, Math.min(1, inputData[i]))
            pcmData[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
          }
        }
        
        // Send to server
        wsRef.current.send(pcmData.buffer)
      }
      
      // Connect audio pipeline
      source.connect(processor)
      processor.connect(audioContext.destination)
      
      setIsListening(true)
      toast.success('Listening...')
      
      // Safety timeout
      recordingTimeout = setTimeout(() => {
        console.log('⏱️ Max recording time reached')
        stopListening()
      }, MAX_RECORDING_MS)
      
    } catch (err) {
      console.error('Failed to start listening:', err)
      setError('Microphone access denied')
      toast.error('Microphone access denied')
    }
  }

  // Stop listening
  const stopListening = () => {
    if (!isListening) return
    
    console.log('🛑 Stopping microphone...')
    
    // Clean up audio
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    
    setIsListening(false)
    setAudioLevel(0)
  }

  // Send greeting request to Oviya
  const requestGreeting = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    
    // Initialize audio context on user interaction (required for autoplay policy)
    if (!playbackContextRef.current) {
      playbackContextRef.current = new AudioContext({ sampleRate: 24000 })
      await playbackContextRef.current.resume()
      console.log('🔊 Audio context initialized')
    }
    
    console.log('👋 Requesting greeting from Oviya...')
    wsRef.current.send(JSON.stringify({
      type: 'greeting',
      text: 'Hello'
    }))
  }, [])

  // Toggle voice mode
  const toggleVoiceMode = useCallback(() => {
    console.log('🔘 Toggle clicked:', { isListening, isSpeaking, isConnected, hasStarted })
    
    if (!isConnected) {
      toast.error('Not connected to server')
      return
    }
    
    // First time: request greeting from Oviya
    if (!hasStarted) {
      setHasStarted(true)
      requestGreeting()
      return
    }
    
    if (isListening) {
      continuousModeRef.current = false
      stopListening()
    } else if (isSpeaking) {
      // Can't interrupt while speaking
      toast('Please wait for response to finish')
    } else {
      continuousModeRef.current = true
      startListening()
    }
  }, [isListening, isSpeaking, isConnected, hasStarted, requestGreeting])

  return {
    isConnected,
    isListening,
    isSpeaking,
    currentEmotion,
    messages,
    audioLevel,
    error,
    toggleVoiceMode
  }
}