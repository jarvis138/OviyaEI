/**
 * Voice Mode Component - Simple and reliable
 */

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, Volume2, Wifi, WifiOff } from 'lucide-react'
import { useVoiceMode } from '@/hooks/useVoiceMode'

export const VoiceMode: React.FC = () => {
  const {
    isConnected,
    isListening,
    isSpeaking,
    currentEmotion,
    messages,
    audioLevel,
    error,
    toggleVoiceMode
  } = useVoiceMode()

  // Determine button state
  const isButtonDisabled = !isConnected || (isSpeaking && messages.length > 0)
  const isFirstInteraction = messages.length === 0
  const buttonLabel = !isConnected ? 'Connecting...' : 
                      isFirstInteraction && !isSpeaking ? 'Talk to Oviya' :
                      isListening ? 'Listening...' : 
                      isSpeaking ? 'Oviya is speaking...' : 
                      'Click to talk'

  // Get last message
  const lastMessage = messages[messages.length - 1]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex flex-col items-center justify-center p-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-white mb-4">
          Oviya Voice <span className="text-sm text-green-400">(Remote WhisperX)</span>
        </h1>
        <p className="text-gray-400">Click the button and speak naturally</p>
      </div>

      {/* Connection Status */}
      <div className="flex items-center gap-2 mb-8">
        {isConnected ? (
          <>
            <Wifi className="w-5 h-5 text-green-500" />
            <span className="text-green-500 text-sm">Connected</span>
          </>
        ) : (
          <>
            <WifiOff className="w-5 h-5 text-red-500" />
            <span className="text-red-500 text-sm">Disconnected</span>
          </>
        )}
      </div>

      {/* Main Button */}
      <div className="relative">
        {/* Pulse animation when active */}
        {(isListening || isSpeaking) && (
          <motion.div
            className="absolute inset-0 rounded-full bg-blue-500 opacity-20"
            animate={{
              scale: [1, 1.5, 1],
              opacity: [0.3, 0, 0.3]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}

        {/* Button */}
        <motion.button
          onClick={toggleVoiceMode}
          disabled={isButtonDisabled}
          className={`
            relative w-40 h-40 rounded-full flex items-center justify-center
            transition-all duration-300 transform
            ${isButtonDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:scale-105'}
            ${isListening ? 'bg-red-500 hover:bg-red-600' : 
              isSpeaking ? 'bg-purple-500' : 
              'bg-blue-500 hover:bg-blue-600'}
          `}
          whileTap={!isButtonDisabled ? { scale: 0.95 } : {}}
        >
          {/* Icon */}
          {isListening ? (
            <MicOff className="w-16 h-16 text-white" />
          ) : isSpeaking ? (
            <Volume2 className="w-16 h-16 text-white" />
          ) : (
            <Mic className="w-16 h-16 text-white" />
          )}
        </motion.button>

        {/* Audio Level Indicator */}
        {isListening && (
          <div className="absolute -bottom-8 left-0 right-0">
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-blue-400"
                animate={{ width: `${Math.min(100, audioLevel * 100)}%` }}
                transition={{ duration: 0.1 }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Status Label */}
      <div className="mt-12 text-center">
        <p className="text-white text-xl">{buttonLabel}</p>
        {error && (
          <p className="text-red-400 text-sm mt-2">{error}</p>
        )}
      </div>

      {/* Messages Display */}
      <div className="mt-12 w-full max-w-2xl">
        <AnimatePresence mode="wait">
          {lastMessage && (
            <motion.div
              key={lastMessage.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-gray-800 rounded-lg p-6"
            >
              <div className="flex items-start gap-4">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  lastMessage.role === 'user' ? 'bg-blue-500' : 'bg-purple-500'
                }`} />
                <div className="flex-1">
                  <p className="text-gray-400 text-xs mb-1">
                    {lastMessage.role === 'user' ? 'You' : 'Oviya'}
                  </p>
                  <p className="text-white">{lastMessage.text}</p>
                  {lastMessage.emotion && (
                    <p className="text-gray-500 text-xs mt-2">
                      Emotion: {lastMessage.emotion.replace(/_/g, ' ')}
                    </p>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Instructions */}
      {!isListening && !isSpeaking && messages.length === 0 && (
        <div className="mt-12 text-center text-gray-500">
          <p className="text-lg mb-4">Click "Talk to Oviya" to start</p>
          <p className="text-sm">Oviya will greet you and begin the conversation</p>
        </div>
      )}
      
      {messages.length > 0 && !isListening && !isSpeaking && (
        <div className="mt-12 text-center text-gray-500">
          <p>Click the button to respond</p>
        </div>
      )}

      {/* Debug Panel - Fixed at bottom */}
      <div className="fixed bottom-4 left-4 right-4 bg-black bg-opacity-80 rounded-lg p-4 text-xs text-gray-400 font-mono max-w-4xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span className="text-gray-500">Status:</span>{' '}
            <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Mode:</span>{' '}
            <span className={isListening ? 'text-blue-400' : isSpeaking ? 'text-purple-400' : 'text-gray-400'}>
              {isListening ? 'Listening' : isSpeaking ? 'Speaking' : 'Idle'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Messages:</span>{' '}
            <span className="text-white">{messages.length}</span>
          </div>
          <div>
            <span className="text-gray-500">Audio Level:</span>{' '}
            <span className="text-white">{(audioLevel * 100).toFixed(0)}%</span>
          </div>
        </div>
        {error && (
          <div className="mt-2 text-red-400">
            Error: {error}
          </div>
        )}
        <div className="mt-2 text-gray-600 text-[10px]">
          💡 Open browser console (F12) for detailed logs
        </div>
      </div>
    </div>
  )
}