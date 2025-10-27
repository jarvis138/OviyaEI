import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, Volume2, VolumeX, Settings, User } from 'lucide-react'
import { useVoiceChat } from '@/hooks/useVoiceChat'
import { useAudioRecorder } from '@/hooks/useAudioRecorder'
import { useAudioPlayer } from '@/hooks/useAudioPlayer'
import { EmotionSelector } from '@/components/EmotionSelector'
import { ConversationHistory } from '@/components/ConversationHistory'
import { ConnectionStatus } from '@/components/ConnectionStatus'
import { toast } from 'react-hot-toast'

export const VoiceChat: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentEmotion, setCurrentEmotion] = useState('empathetic')
  const [showSettings, setShowSettings] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  
  const {
    isConnected,
    sessionId,
    messages,
    isAiSpeaking,
    connect,
    disconnect,
    sendMessage,
    interrupt
  } = useVoiceChat()
  
  const {
    startRecording,
    stopRecording,
    isRecording: isRecorderActive,
    audioBlob
  } = useAudioRecorder()
  
  const {
    playAudio,
    stopAudio,
    isPlaying: isPlayerActive
  } = useAudioPlayer()
  
  // Handle recording state
  const handleStartRecording = async () => {
    if (!isConnected) {
      toast.error('Please connect first')
      return
    }
    
    if (isAiSpeaking) {
      await interrupt()
    }
    
    setIsRecording(true)
    await startRecording()
  }
  
  const handleStopRecording = async () => {
    setIsRecording(false)
    const blob = await stopRecording()
    
    if (blob && blob.size > 0) {
      // Convert blob to audio data and send
      const arrayBuffer = await blob.arrayBuffer()
      const audioData = new Uint8Array(arrayBuffer)
      
      await sendMessage({
        type: 'audio',
        data: audioData,
        emotion: currentEmotion
      })
    }
  }
  
  // Handle AI response audio
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      if (lastMessage.role === 'assistant' && lastMessage.audioChunks) {
        // Play AI response audio
        const audioData = new Uint8Array(
          lastMessage.audioChunks.map(chunk => 
            new Uint8Array(Buffer.from(chunk, 'hex'))
          ).flat()
        )
        
        playAudio(audioData)
      }
    }
  }, [messages, playAudio])
  
  // Connect on mount
  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])
  
  return (
    <div className="max-w-4xl mx-auto">
      {/* Connection Status */}
      <ConnectionStatus 
        isConnected={isConnected} 
        sessionId={sessionId}
        isAiSpeaking={isAiSpeaking}
      />
      
      {/* Main Chat Interface */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        {/* Emotion Selector */}
        <div className="mb-6">
          <EmotionSelector
            currentEmotion={currentEmotion}
            onEmotionChange={setCurrentEmotion}
            disabled={isRecording || isAiSpeaking}
          />
        </div>
        
        {/* Voice Interface */}
        <div className="flex flex-col items-center space-y-6">
          {/* Mic Button */}
          <motion.button
            className={`w-24 h-24 rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-300 ${
              isRecording 
                ? 'bg-red-500 hover:bg-red-600 scale-110' 
                : 'bg-purple-500 hover:bg-purple-600'
            }`}
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            disabled={!isConnected || isPlayerActive}
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
          <div className="text-center">
            <AnimatePresence mode="wait">
              {isRecording ? (
                <motion.p
                  key="recording"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-lg text-red-600 font-medium"
                >
                  Listening... Click to stop
                </motion.p>
              ) : isAiSpeaking ? (
                <motion.p
                  key="ai-speaking"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-lg text-purple-600 font-medium"
                >
                  Oviya is speaking...
                </motion.p>
              ) : (
                <motion.p
                  key="ready"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-lg text-gray-600 font-medium"
                >
                  Click to start talking
                </motion.p>
              )}
            </AnimatePresence>
          </div>
          
          {/* Control Buttons */}
          <div className="flex space-x-4">
            <button
              className={`p-3 rounded-full transition-colors ${
                isPlayerActive 
                  ? 'bg-red-100 text-red-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              onClick={stopAudio}
              disabled={!isPlayerActive}
            >
              {isPlayerActive ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
            
            <button
              className="p-3 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
              onClick={() => setShowHistory(!showHistory)}
            >
              <User className="w-5 h-5" />
            </button>
            
            <button
              className="p-3 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
      
      {/* Conversation History */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-white rounded-2xl shadow-xl p-6 mb-8"
          >
            <ConversationHistory messages={messages} />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-white rounded-2xl shadow-xl p-6"
          >
            <h3 className="text-lg font-semibold mb-4">Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Audio Quality
                </label>
                <select className="w-full p-2 border border-gray-300 rounded-lg">
                  <option value="high">High Quality (24kHz)</option>
                  <option value="medium">Medium Quality (16kHz)</option>
                  <option value="low">Low Quality (8kHz)</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Auto-play Responses
                </label>
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Show Transcriptions
                </label>
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}


