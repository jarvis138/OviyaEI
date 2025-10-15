import { useState, useEffect, useRef } from 'react'
import { toast } from 'react-hot-toast'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  audioChunks?: string[]
  timestamp: number
  emotion?: string
  wordTimestamps?: Array<{ word: string; start: number; end: number }>
}

interface LatencyMetrics {
  sttLatency: number
  totalLatency: number
  lastUpdate: number
}

interface LiveDemoState {
  isConnected: boolean
  isRecording: boolean
  isAiSpeaking: boolean
  currentEmotion: string | null
  messages: Message[]
  audioLevel: number
  latencyMetrics: LatencyMetrics
  continuousMode: boolean
  startRecording: () => Promise<void>
  stopRecording: () => Promise<void>
  interrupt: () => Promise<void>
  sendTextMessage: (text: string) => Promise<void>
  toggleContinuousMode: () => void
}

export const useLiveDemo = (): LiveDemoState => {
  const [isConnected, setIsConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isAiSpeaking, setIsAiSpeaking] = useState(false)
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [audioLevel, setAudioLevel] = useState(0)
  const [continuousMode, setContinuousMode] = useState(true) // Always on by default like ChatGPT
  const [latencyMetrics, setLatencyMetrics] = useState<LatencyMetrics>({
    sttLatency: 0,
    totalLatency: 0,
    lastUpdate: 0
  })
  
  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const playbackContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const requestStartTimeRef = useRef<number>(0)
  const audioQueueRef = useRef<AudioBuffer[]>([])
  const isPlayingRef = useRef(false)
  const silenceSinceRef = useRef<number | null>(null)
  const maxRecordTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  
  // Connect to WebSocket backend
  useEffect(() => {
    const connectToBackend = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/conversation?user_id=web_user')
        
        ws.onopen = () => {
          console.log('✅ Connected to Oviya backend')
          setIsConnected(true)
          toast.success('Connected to Oviya AI')
          wsRef.current = ws
        }
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            console.log('📨 Received:', data.type)
            
            switch (data.type) {
              case 'transcription':
                // Calculate STT latency
                const sttLatency = performance.now() - requestStartTimeRef.current
                setLatencyMetrics(prev => ({
                  ...prev,
                  sttLatency,
                  lastUpdate: Date.now()
                }))
                
                // User's speech was transcribed
                const userMessage: Message = {
                  id: Date.now().toString(),
                  role: 'user',
                  text: data.text,
                  timestamp: Date.now(),
                  emotion: data.emotion,
                  wordTimestamps: data.word_timestamps
                }
                setMessages(prev => [...prev, userMessage])
                break
                
              case 'response':
                // Calculate total latency
                const totalLatency = performance.now() - requestStartTimeRef.current
                setLatencyMetrics(prev => ({
                  ...prev,
                  totalLatency,
                  lastUpdate: Date.now()
                }))
                
                // AI response received
                const assistantMessage: Message = {
                  id: Date.now().toString(),
                  role: 'assistant',
                  text: data.text,
                  audioChunks: data.audio_chunks,
                  timestamp: Date.now(),
                  emotion: data.emotion
                }
                setMessages(prev => [...prev, assistantMessage])
                setIsAiSpeaking(true)
                setCurrentEmotion(data.emotion)
                
                // Play audio if available
                if (data.audio_chunks && data.audio_chunks.length > 0) {
                  playAudioWithBuffer(data.audio_chunks)
                } else {
                  setIsAiSpeaking(false)
                }
                break
                
              case 'error':
                console.error('Backend error:', data.message)
                toast.error(data.message)
                setIsAiSpeaking(false)
                break
            }
          } catch (error) {
            console.error('Error parsing message:', error)
          }
        }
        
        ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error)
          toast.error('Connection error')
        }
        
        ws.onclose = () => {
          console.log('❌ Disconnected from backend')
          setIsConnected(false)
          wsRef.current = null
          
          // Attempt to reconnect after 3 seconds
          setTimeout(() => {
            console.log('🔄 Attempting to reconnect...')
            connectToBackend()
          }, 3000)
        }
      } catch (error) {
        console.error('Failed to connect:', error)
        toast.error('Failed to connect to backend')
      }
    }
    
    connectToBackend()
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])
  
  // Continuous mode: auto-resume listening after response
  useEffect(() => {
    if (continuousMode && !isAiSpeaking && !isRecording && isConnected && messages.length > 0) {
      const timer = setTimeout(() => {
        console.log('🔄 Auto-resuming listening (continuous mode)')
        startRecording()
      }, 800)
      
      return () => clearTimeout(timer)
    }
  }, [isAiSpeaking, continuousMode, isConnected, messages.length])
  
  // Audio level monitoring
  useEffect(() => {
    if (!analyserRef.current || !isRecording) return
    
    const analyser = analyserRef.current
    const dataArray = new Uint8Array(analyser.frequencyBinCount)
    
    const updateLevel = () => {
      analyser.getByteFrequencyData(dataArray)
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length
      setAudioLevel(average / 255)
      
      // Silence auto-stop: if below threshold for 800ms, stop recording
      const threshold = 0.06 // empirical
      const now = Date.now()
      if ((average / 255) < threshold) {
        if (silenceSinceRef.current === null) {
          silenceSinceRef.current = now
        } else if (now - silenceSinceRef.current > 800 && isRecording) {
          stopRecording()
        }
      } else {
        silenceSinceRef.current = null
      }

      if (isRecording) {
        requestAnimationFrame(updateLevel)
      }
    }
    
    updateLevel()
  }, [isRecording])
  
  const startRecording = async () => {
    if (!isConnected) {
      toast.error('Not connected to backend')
      return
    }
    
    if (isAiSpeaking) {
      await interrupt()
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      })
      
      streamRef.current = stream
      audioContextRef.current = new AudioContext({ sampleRate: 16000 })
      const source = audioContextRef.current.createMediaStreamSource(stream)
      
      // Create analyser for audio level visualization
      const analyser = audioContextRef.current.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.8
      analyserRef.current = analyser
      
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1)
      
      processor.onaudioprocess = (e) => {
        const ws = wsRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN) return
        
        // Get input float32 samples at the device/sample rate
        const inputBuffer = e.inputBuffer.getChannelData(0)
        const inputSampleRate = audioContextRef.current?.sampleRate || 48000
        
        // Downsample to 16kHz expected by backend
        const pcm16 = downsampleTo16kPCM(inputBuffer, inputSampleRate)
        if (!pcm16 || pcm16.length === 0) return
        
        // Simple energy gate to avoid spamming near-silence
        let energy = 0
        for (let i = 0; i < pcm16.length; i += 64) {
          const s = pcm16[i]
          energy += Math.abs(s)
        }
        if (energy / (pcm16.length / 64) < 50) {
          // Very low energy; still send some silence but less frequently
          // Optionally skip; for robustness we still send
        }
        
        ws.send(pcm16.buffer)
      }
      
      source.connect(analyser)
      analyser.connect(processor)
      processor.connect(audioContextRef.current.destination)
      processorRef.current = processor
      
      // Mark request start time for latency tracking
      requestStartTimeRef.current = performance.now()
      
      setIsRecording(true)
      // Do not mark AI as speaking yet; wait until playback actually starts
      toast.success('Listening...')

      // Safety: auto-stop after 12 seconds max utterance
      if (maxRecordTimeoutRef.current) clearTimeout(maxRecordTimeoutRef.current)
      maxRecordTimeoutRef.current = setTimeout(() => {
        if (isRecording) stopRecording()
      }, 12000)
      
    } catch (error) {
      console.error('Error starting recording:', error)
      toast.error('Failed to access microphone')
    }
  }
  
  const stopRecording = async () => {
    if (maxRecordTimeoutRef.current) {
      clearTimeout(maxRecordTimeoutRef.current)
      maxRecordTimeoutRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    analyserRef.current = null
    processorRef.current = null
    setIsRecording(false)
    setAudioLevel(0)
    silenceSinceRef.current = null
  }
  
  // Utilities
  function downsampleTo16kPCM(input: Float32Array, inputSampleRate: number): Int16Array {
    const targetRate = 16000
    if (inputSampleRate === targetRate) {
      const out = new Int16Array(input.length)
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]))
        out[i] = s < 0 ? s * 32768 : s * 32767
      }
      return out
    }
    // Calculate ratio
    const ratio = inputSampleRate / targetRate
    const newLength = Math.floor(input.length / ratio)
    const result = new Int16Array(newLength)
    let idx = 0
    let i = 0
    while (idx < newLength) {
      // Simple averaging downsample
      const nextIdx = Math.floor((idx + 1) * ratio)
      let sum = 0
      let count = 0
      for (; i < nextIdx && i < input.length; i++) {
        sum += input[i]
        count++
      }
      const sample = count > 0 ? sum / count : 0
      const s = Math.max(-1, Math.min(1, sample))
      result[idx] = s < 0 ? s * 32768 : s * 32767
      idx++
    }
    return result
  }
  
  const sendTextMessage = async (text: string) => {
    // For now, text messages are not supported by the backend
    toast.error('Text messages not yet supported. Please use voice.')
  }
  
  const interrupt = async () => {
    if (isRecording) {
      await stopRecording()
    }
    setIsAiSpeaking(false)
    audioQueueRef.current = []
    isPlayingRef.current = false
  }
  
  const playAudioWithBuffer = async (audioChunks: string[]) => {
    try {
      // Reuse a single playback context to avoid autoplay restrictions
      if (!playbackContextRef.current) {
        playbackContextRef.current = new AudioContext({ sampleRate: 24000 })
      }
      const audioContext = playbackContextRef.current
      if (audioContext.state === 'suspended') {
        try { await audioContext.resume() } catch {}
      }
      
      // Decode all chunks
      for (const chunk of audioChunks) {
        try {
          const audioData = Uint8Array.from(atob(chunk), c => c.charCodeAt(0))
          const int16Data = new Int16Array(audioData.buffer)
          const float32Data = new Float32Array(int16Data.length)
          
          for (let i = 0; i < int16Data.length; i++) {
            float32Data[i] = int16Data[i] / 32768.0
          }
          
          const audioBuffer = audioContext.createBuffer(1, float32Data.length, 24000)
          audioBuffer.getChannelData(0).set(float32Data)
          
          audioQueueRef.current.push(audioBuffer)
        } catch (e) {
          console.error('Error decoding chunk:', e)
        }
      }
      
      // Play buffered audio
      if (!isPlayingRef.current && audioQueueRef.current.length > 0) {
        // Mark AI as speaking when playback starts
        setIsAiSpeaking(true)
        playNextInQueue(audioContext)
      }
      
    } catch (error) {
      console.error('Error processing audio response:', error)
      setIsAiSpeaking(false)
    }
  }
  
  const playNextInQueue = (audioContext: AudioContext) => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false
      setIsAiSpeaking(false)
      return
    }
    
    isPlayingRef.current = true
    const buffer = audioQueueRef.current.shift()!
    
    const source = audioContext.createBufferSource()
    source.buffer = buffer
    source.connect(audioContext.destination)
    
    source.onended = () => {
      playNextInQueue(audioContext)
    }
    
    source.start(audioContext.currentTime)
  }
  
  const toggleContinuousMode = () => {
    setContinuousMode(prev => !prev)
    toast.success(continuousMode ? 'Continuous mode OFF' : 'Continuous mode ON')
  }
  
  return {
    isConnected,
    isRecording,
    isAiSpeaking,
    currentEmotion,
    messages,
    audioLevel,
    latencyMetrics,
    continuousMode,
    startRecording,
    stopRecording,
    interrupt,
    sendTextMessage,
    toggleContinuousMode
  }
}