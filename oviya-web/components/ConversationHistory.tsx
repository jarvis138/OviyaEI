import React from 'react'
import { motion } from 'framer-motion'
import { User, Bot, Clock } from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  audioChunks?: string[]
  timestamp: number
  emotion?: string
}

interface ConversationHistoryProps {
  messages: Message[]
}

export const ConversationHistory: React.FC<ConversationHistoryProps> = ({ messages }) => {
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }
  
  const getEmotionColor = (emotion?: string) => {
    const colors = {
      empathetic: 'text-pink-500',
      encouraging: 'text-yellow-500',
      calm: 'text-blue-500',
      joyful: 'text-green-500',
      concerned: 'text-orange-500'
    }
    return colors[emotion as keyof typeof colors] || 'text-gray-500'
  }
  
  if (messages.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 mb-2">
          <Bot className="w-12 h-12 mx-auto" />
        </div>
        <p className="text-gray-500">No conversation yet</p>
        <p className="text-sm text-gray-400 mt-1">
          Start talking to begin your conversation with Oviya
        </p>
      </div>
    )
  }
  
  return (
    <div className="space-y-4 max-h-96 overflow-y-auto">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Conversation History
      </h3>
      
      {messages.map((message, index) => (
        <motion.div
          key={message.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div className={`
            max-w-xs lg:max-w-md px-4 py-3 rounded-2xl
            ${message.role === 'user' 
              ? 'bg-purple-500 text-white' 
              : 'bg-gray-100 text-gray-900'
            }
          `}>
            {/* Message Header */}
            <div className="flex items-center space-x-2 mb-2">
              {message.role === 'user' ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
              
              <span className="text-xs font-medium">
                {message.role === 'user' ? 'You' : 'Oviya'}
              </span>
              
              {message.emotion && message.role === 'assistant' && (
                <span className={`text-xs ${getEmotionColor(message.emotion)}`}>
                  • {message.emotion}
                </span>
              )}
              
              <span className="text-xs opacity-70">
                <Clock className="w-3 h-3 inline mr-1" />
                {formatTime(message.timestamp)}
              </span>
            </div>
            
            {/* Message Content */}
            <div className="text-sm">
              {message.text}
            </div>
            
            {/* Audio Indicator */}
            {message.audioChunks && message.audioChunks.length > 0 && (
              <div className="mt-2 flex items-center space-x-1">
                <div className="w-2 h-2 bg-current rounded-full animate-pulse"></div>
                <span className="text-xs opacity-70">
                  {message.audioChunks.length} audio chunks
                </span>
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  )
}


