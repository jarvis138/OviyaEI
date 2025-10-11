import React, { useState, useEffect, useRef } from 'react'
import { io, Socket } from 'socket.io-client'
import { toast } from 'react-hot-toast'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  audioChunks?: string[]
  timestamp: number
  emotion?: string
}

interface VoiceChatState {
  isConnected: boolean
  sessionId: string | null
  messages: Message[]
  isAiSpeaking: boolean
  connect: () => void
  disconnect: () => void
  sendMessage: (data: any) => Promise<void>
  interrupt: () => Promise<void>
}

export const useVoiceChat = (): VoiceChatState => {
  const [isConnected, setIsConnected] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [isAiSpeaking, setIsAiSpeaking] = useState(false)
  
  const socketRef = useRef<Socket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  
  const connect = () => {
    if (socketRef.current?.connected) return
    
    const socket = io(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || 'http://localhost:8002', {
      transports: ['websocket'],
      timeout: 10000,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    })
    
    socketRef.current = socket
    
    // Connection events
    socket.on('connect', () => {
      setIsConnected(true)
      toast.success('Connected to Oviya AI')
      
      // Create session
      socket.emit('create_session', { user_id: 'web_user' })
    })
    
    socket.on('disconnect', () => {
      setIsConnected(false)
      toast.error('Disconnected from Oviya AI')
    })
    
    socket.on('connect_error', (error) => {
      console.error('Connection error:', error)
      toast.error('Failed to connect to Oviya AI')
    })
    
    // Session events
    socket.on('session_created', (data) => {
      setSessionId(data.session_id)
      console.log('Session created:', data.session_id)
    })
    
    // Message events
    socket.on('message_response', (data) => {
      const message: Message = {
        id: data.request_id || Date.now().toString(),
        role: 'assistant',
        text: data.text,
        audioChunks: data.audio_chunks,
        timestamp: Date.now(),
        emotion: data.emotion
      }
      
      setMessages(prev => [...prev, message])
      setIsAiSpeaking(false)
      
      toast.success('Response received')
    })
    
    socket.on('interrupt_result', (data) => {
      if (data.interrupted) {
        setIsAiSpeaking(false)
        toast.success('Interrupted AI response')
      }
    })
    
    socket.on('error', (error) => {
      console.error('Socket error:', error)
      toast.error(error.message || 'An error occurred')
    })
  }
  
  const disconnect = () => {
    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    setIsConnected(false)
    setSessionId(null)
    setIsAiSpeaking(false)
  }
  
  const sendMessage = async (data: any) => {
    if (!socketRef.current?.connected || !sessionId) {
      toast.error('Not connected to Oviya AI')
      return
    }
    
    try {
      // Add user message to history
      const userMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        text: data.text || '[Audio Message]',
        timestamp: Date.now(),
        emotion: data.emotion
      }
      
      setMessages(prev => [...prev, userMessage])
      setIsAiSpeaking(true)
      
      // Send message to orchestrator
      socketRef.current.emit('send_message', {
        session_id: sessionId,
        text: data.text,
        emotion: data.emotion || 'empathetic',
        priority: 'normal'
      })
      
    } catch (error) {
      console.error('Error sending message:', error)
      toast.error('Failed to send message')
      setIsAiSpeaking(false)
    }
  }
  
  const interrupt = async () => {
    if (!socketRef.current?.connected || !sessionId) {
      return
    }
    
    try {
      socketRef.current.emit('interrupt', {
        session_id: sessionId
      })
      
      setIsAiSpeaking(false)
      
    } catch (error) {
      console.error('Error interrupting:', error)
      toast.error('Failed to interrupt')
    }
  }
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [])
  
  return {
    isConnected,
    sessionId,
    messages,
    isAiSpeaking,
    connect,
    disconnect,
    sendMessage,
    interrupt
  }
}


