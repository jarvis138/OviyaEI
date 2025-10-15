import React from 'react'
import { motion } from 'framer-motion'
import { Mic, MicOff, Volume2, VolumeX, Brain, Heart } from 'lucide-react'
import { useLiveDemo } from '@/hooks/useLiveDemo'

export const LiveAIDemo: React.FC = () => {
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

  return (
    <section id="demo" className="py-20 bg-white">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-light text-gray-900 mb-4">
            Experience AI That Understands
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Have a real conversation with Oviya right now. No signup required.
          </p>
        </motion.div>

        {/* Demo Interface */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          viewport={{ once: true }}
          className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-3xl shadow-xl p-8 md:p-12 max-w-4xl mx-auto"
        >
          {/* Connection Status */}
          <div className="flex items-center justify-center mb-8">
            <div className={`w-3 h-3 rounded-full mr-3 ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`} />
            <span className="text-sm text-gray-600">
              {isConnected ? 'Connected to Oviya' : 'Connecting...'}
            </span>
          </div>

          {/* Emotion Indicator */}
          {currentEmotion && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center justify-center mb-6"
            >
              <div className="bg-white rounded-full px-4 py-2 shadow-sm">
                <span className="text-sm text-gray-600">
                  Detected: <span className="font-medium text-purple-600">{currentEmotion}</span>
                </span>
              </div>
            </motion.div>
          )}

          {/* Voice Interface */}
          <div className="text-center mb-8">
            <motion.button
              className={`w-24 h-24 rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-300 ${
                isRecording 
                  ? 'bg-red-500 hover:bg-red-600 scale-110' 
                  : 'bg-purple-500 hover:bg-purple-600'
              }`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!isConnected || isAiSpeaking}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              animate={isRecording ? { scale: [1, 1.1, 1] } : {}}
              transition={{ duration: 1, repeat: isRecording ? Infinity : 0 }}
            >
              {isRecording ? (
                <MicOff className="w-8 h-8" />
              ) : (
                <Mic className="w-8 h-8" />
              )}
            </motion.button>

            {/* Status Text */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6"
            >
              {isRecording ? (
                <p className="text-lg text-red-600 font-medium">
                  Listening... Click to stop
                </p>
              ) : isAiSpeaking ? (
                <p className="text-lg text-purple-600 font-medium">
                  Oviya is speaking...
                </p>
              ) : (
                <p className="text-lg text-gray-600 font-medium">
                  Click to start talking
                </p>
              )}
            </motion.div>
          </div>

          {/* Control Buttons */}
          <div className="flex justify-center space-x-4 mb-8">
            <button
              className={`p-3 rounded-full transition-colors ${
                isAiSpeaking 
                  ? 'bg-red-100 text-red-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              onClick={interrupt}
              disabled={!isAiSpeaking}
            >
              {isAiSpeaking ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
          </div>

          {/* Recent Messages */}
          {messages.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-2xl p-6 shadow-sm"
            >
              <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-600" />
                Recent Conversation
              </h3>
              <div className="space-y-3 max-h-40 overflow-y-auto">
                {messages.slice(-3).map((message, index) => (
                  <div key={index} className={`flex items-start gap-3 ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}>
                    <div className={`max-w-xs px-4 py-2 rounded-2xl ${
                      message.role === 'user' 
                        ? 'bg-purple-100 text-purple-900' 
                        : 'bg-gray-100 text-gray-900'
                    }`}>
                      <p className="text-sm">{message.text}</p>
                      {message.emotion && (
                        <div className="flex items-center gap-1 mt-1">
                          <Heart className="w-3 h-3 text-red-500" />
                          <span className="text-xs text-gray-500">{message.emotion}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </section>
  )
}
