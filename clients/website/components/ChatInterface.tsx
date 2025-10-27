import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, Send, Heart, Sparkles, Volume2, VolumeX } from 'lucide-react'
import { useLiveDemo } from '@/hooks/useLiveDemo'

export const ChatInterface: React.FC = () => {
  const {
    isConnected,
    isRecording,
    isAiSpeaking,
    currentEmotion,
    messages,
    startRecording,
    stopRecording,
    interrupt
  } = useLiveDemo()

  const [textInput, setTextInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendText = () => {
    if (textInput.trim()) {
      // Handle text message sending
      setTextInput('')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 flex items-center justify-center p-4">
      {/* Main Chat Container - Sesame Style */}
      <div className="w-full max-w-3xl h-[85vh] bg-white rounded-3xl shadow-2xl flex flex-col overflow-hidden">
        
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
              <Heart className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h1 className="text-white font-semibold text-lg">Oviya</h1>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-gray-400'}`} />
                <span className="text-white/80 text-xs">
                  {isConnected ? 'Connected' : 'Connecting...'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Emotion Indicator */}
          {currentEmotion && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white/20 backdrop-blur-sm rounded-full px-4 py-2"
            >
              <span className="text-white text-sm flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                {currentEmotion}
              </span>
            </motion.div>
          )}
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                <div className="w-20 h-20 bg-gradient-to-br from-purple-100 to-blue-100 rounded-full flex items-center justify-center mb-6">
                  <Heart className="w-10 h-10 text-purple-600" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-3">
                  Hi, I'm Oviya
                </h2>
                <p className="text-gray-600 max-w-md mb-6">
                  Your empathetic AI companion. I understand emotions and remember our conversations. 
                  Let's talk!
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-500">
                  <span className="bg-purple-50 px-3 py-1 rounded-full">Real-time Voice</span>
                  <span className="bg-blue-50 px-3 py-1 rounded-full">49 Emotions</span>
                  <span className="bg-indigo-50 px-3 py-1 rounded-full">Persistent Memory</span>
                </div>
              </motion.div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[75%] ${
                    message.role === 'user' 
                      ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white' 
                      : 'bg-gray-100 text-gray-900'
                  } rounded-2xl px-5 py-3 shadow-sm`}>
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    {message.emotion && (
                      <div className="flex items-center gap-1 mt-2 opacity-70">
                        <Sparkles className="w-3 h-3" />
                        <span className="text-xs">{message.emotion}</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
              <div ref={messagesEndRef} />
            </>
          )}

          {/* AI Speaking Indicator */}
          {isAiSpeaking && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="bg-gray-100 rounded-2xl px-5 py-3">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-sm text-gray-600">Oviya is thinking...</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <div className="flex items-center gap-3">
            {/* Voice Button */}
            <motion.button
              className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                isRecording 
                  ? 'bg-red-500 hover:bg-red-600 text-white scale-110' 
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white'
              }`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!isConnected || isAiSpeaking}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              animate={isRecording ? { scale: [1, 1.1, 1] } : {}}
              transition={{ duration: 1, repeat: isRecording ? Infinity : 0 }}
            >
              {isRecording ? (
                <MicOff className="w-5 h-5" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </motion.button>

            {/* Text Input */}
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendText()}
              placeholder={isRecording ? "Listening..." : "Type a message or use voice..."}
              disabled={isRecording || isAiSpeaking}
              className="flex-1 px-4 py-3 bg-white border border-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 placeholder-gray-400"
            />

            {/* Send Button */}
            <button
              onClick={handleSendText}
              disabled={!textInput.trim() || isRecording || isAiSpeaking}
              className="flex-shrink-0 w-12 h-12 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-full flex items-center justify-center transition-all"
            >
              <Send className="w-5 h-5" />
            </button>

            {/* Volume Control */}
            <button
              onClick={interrupt}
              disabled={!isAiSpeaking}
              className="flex-shrink-0 w-12 h-12 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed text-gray-700 rounded-full flex items-center justify-center transition-all"
            >
              {isAiSpeaking ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
          </div>

          {/* Status Text */}
          <div className="mt-2 text-center">
            <AnimatePresence mode="wait">
              {isRecording ? (
                <motion.p
                  key="recording"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm text-red-600 font-medium"
                >
                  🎤 Listening... Click mic to stop
                </motion.p>
              ) : isAiSpeaking ? (
                <motion.p
                  key="speaking"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm text-purple-600 font-medium"
                >
                  🔊 Oviya is speaking...
                </motion.p>
              ) : (
                <motion.p
                  key="ready"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-sm text-gray-500"
                >
                  Click mic to talk or type a message
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}
