import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { VoiceOrb } from './VoiceOrb'
import { useLiveDemo } from '@/hooks/useLiveDemo'
import { Sparkles, Heart, Activity } from 'lucide-react'

// Emotion to color mapping
const emotionColors: Record<string, string> = {
  'joyful_excited': 'from-yellow-900 via-orange-900 to-red-900',
  'playful': 'from-pink-900 via-purple-900 to-indigo-900',
  'calm_supportive': 'from-blue-900 via-cyan-900 to-teal-900',
  'empathetic_sad': 'from-indigo-900 via-purple-900 to-pink-900',
  'thoughtful': 'from-slate-900 via-gray-900 to-zinc-900',
  'encouraging': 'from-green-900 via-emerald-900 to-teal-900',
  'gentle_caring': 'from-purple-900 via-violet-900 to-fuchsia-900',
  'neutral': 'from-purple-900 via-blue-900 to-indigo-900'
}

const getEmotionColor = (emotion: string | null): string => {
  if (!emotion) return emotionColors.neutral
  
  // Try exact match first
  if (emotionColors[emotion]) return emotionColors[emotion]
  
  // Try partial match
  for (const [key, value] of Object.entries(emotionColors)) {
    if (emotion.includes(key) || key.includes(emotion.split('_')[0])) {
      return value
    }
  }
  
  return emotionColors.neutral
}

export const ChatGPTVoiceMode: React.FC = () => {
  const {
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
    toggleContinuousMode
  } = useLiveDemo()
  
  const [displayText, setDisplayText] = useState('')
  const [showEmotion, setShowEmotion] = useState(false)
  const [showMetrics, setShowMetrics] = useState(false)
  const [backgroundGradient, setBackgroundGradient] = useState(emotionColors.neutral)
  
  // Update display text based on latest message with word animation
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      
      // Animate text appearance
      if (lastMessage.wordTimestamps && lastMessage.wordTimestamps.length > 0) {
        // Word-by-word animation
        let currentIndex = 0
        setDisplayText('')
        
        const animateWords = () => {
          if (currentIndex < lastMessage.wordTimestamps!.length) {
            setDisplayText(prev => prev + lastMessage.wordTimestamps![currentIndex].word + ' ')
            currentIndex++
            setTimeout(animateWords, 50)
          }
        }
        
        animateWords()
      } else {
        // Instant display if no timestamps
        setDisplayText(lastMessage.text)
      }
      
      // Show emotion briefly
      if (lastMessage.emotion) {
        setShowEmotion(true)
        setTimeout(() => setShowEmotion(false), 3000)
      }
    }
  }, [messages])
  
  // Update background gradient based on emotion
  useEffect(() => {
    if (currentEmotion) {
      const newGradient = getEmotionColor(currentEmotion)
      setBackgroundGradient(newGradient)
    }
  }, [currentEmotion])
  
  const handleOrbClick = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }
  
  const getStatusText = () => {
    if (!isConnected) return 'Connecting to Oviya...'
    if (isRecording) return 'Listening...'
    if (isAiSpeaking) return 'Oviya is speaking...'
    return 'Tap to talk'
  }
  
  const getStatusColor = () => {
    if (!isConnected) return 'text-gray-400'
    if (isRecording) return 'text-blue-400'
    if (isAiSpeaking) return 'text-purple-400'
    return 'text-gray-300'
  }
  
  return (
    <div className={`min-h-screen bg-gradient-to-br ${backgroundGradient} flex flex-col items-center justify-center p-6 overflow-hidden transition-all duration-1000`}>
      {/* Background animated gradient orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            x: [0, 50, 0],
            y: [0, 30, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            x: [0, -50, 0],
            y: [0, -30, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1
          }}
        />
      </div>
      
      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center max-w-4xl w-full">
        {/* Logo/Title */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 flex items-center gap-3"
        >
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-600 rounded-full flex items-center justify-center">
            <Heart className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-4xl font-light text-white">Oviya</h1>
        </motion.div>
        
        {/* Voice Orb */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <VoiceOrb
            isListening={isRecording}
            isSpeaking={isAiSpeaking}
            isConnected={isConnected}
            audioLevel={audioLevel}
            onClick={handleOrbClick}
          />
        </motion.div>
        
        {/* Status Text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-8 text-center"
        >
          <AnimatePresence mode="wait">
            <motion.p
              key={getStatusText()}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className={`text-2xl font-light ${getStatusColor()}`}
            >
              {getStatusText()}
            </motion.p>
          </AnimatePresence>
        </motion.div>
        
        {/* Transcript Display */}
        <AnimatePresence>
          {displayText && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-8 max-w-2xl w-full"
            >
              <div className="bg-white/10 backdrop-blur-md rounded-3xl px-8 py-6 shadow-2xl">
                <p className="text-white text-lg leading-relaxed text-center">
                  {displayText}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Emotion Indicator */}
        <AnimatePresence>
          {showEmotion && currentEmotion && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="mt-6"
            >
              <div className="bg-purple-500/20 backdrop-blur-sm rounded-full px-6 py-3 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-300" />
                <span className="text-purple-200 text-sm font-medium">
                  {currentEmotion.replace(/_/g, ' ')}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Conversation History Count */}
        {messages.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-12 text-gray-400 text-sm"
          >
            {messages.length} {messages.length === 1 ? 'exchange' : 'exchanges'} in this conversation
          </motion.div>
        )}
        
        {/* Instructions */}
        {!isRecording && !isAiSpeaking && isConnected && messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="mt-16 text-center max-w-md"
          >
            <p className="text-gray-400 text-sm leading-relaxed">
              Click the orb to start a voice conversation with Oviya. 
              She'll understand your emotions and respond with empathy.
            </p>
          </motion.div>
        )}
      </div>
      
      {/* Control Panel */}
      <div className="absolute bottom-8 left-8 flex flex-col gap-3">
        {/* Metrics Toggle */}
        <motion.button
          onClick={() => setShowMetrics(!showMetrics)}
          className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm text-gray-400 hover:bg-white/20 transition-all"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Activity className="w-4 h-4" />
          <span className="text-xs font-medium">Metrics</span>
        </motion.button>
      </div>
      
      {/* Latency Metrics Display */}
      <AnimatePresence>
        {showMetrics && latencyMetrics.lastUpdate > 0 && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="absolute bottom-8 left-8 mt-20 bg-black/50 backdrop-blur-md rounded-2xl px-6 py-4 text-white"
          >
            <h3 className="text-sm font-semibold mb-3 text-gray-300">Performance Metrics</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between gap-8">
                <span className="text-gray-400">STT Latency:</span>
                <span className="font-mono text-green-400">{latencyMetrics.sttLatency.toFixed(0)}ms</span>
              </div>
              <div className="flex justify-between gap-8">
                <span className="text-gray-400">Total Latency:</span>
                <span className="font-mono text-blue-400">{latencyMetrics.totalLatency.toFixed(0)}ms</span>
              </div>
              <div className="flex justify-between gap-8">
                <span className="text-gray-400">Audio Level:</span>
                <span className="font-mono text-purple-400">{(audioLevel * 100).toFixed(0)}%</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Keyboard shortcut hint */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="absolute bottom-8 right-8 text-gray-500 text-xs flex items-center gap-2"
      >
        <span>Press</span>
        <kbd className="px-2 py-1 bg-white/10 rounded">Space</kbd>
        <span>to talk</span>
      </motion.div>
    </div>
  )
}