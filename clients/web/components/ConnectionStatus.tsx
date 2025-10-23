import React from 'react'
import { motion } from 'framer-motion'
import { Wifi, WifiOff, Mic, Volume2 } from 'lucide-react'

interface ConnectionStatusProps {
  isConnected: boolean
  sessionId: string | null
  isAiSpeaking: boolean
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  sessionId,
  isAiSpeaking
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-lg p-4 mb-6"
    >
      <div className="flex items-center justify-between">
        {/* Connection Status */}
        <div className="flex items-center space-x-3">
          <div className={`
            w-3 h-3 rounded-full
            ${isConnected ? 'bg-green-500' : 'bg-red-500'}
          `} />
          
          <div>
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <Wifi className="w-4 h-4 text-green-500" />
              ) : (
                <WifiOff className="w-4 h-4 text-red-500" />
              )}
              
              <span className={`text-sm font-medium ${
                isConnected ? 'text-green-700' : 'text-red-700'
              }`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {sessionId && (
              <p className="text-xs text-gray-500 mt-1">
                Session: {sessionId.slice(0, 8)}...
              </p>
            )}
          </div>
        </div>
        
        {/* AI Speaking Status */}
        <div className="flex items-center space-x-2">
          {isAiSpeaking ? (
            <>
              <Volume2 className="w-4 h-4 text-purple-500" />
              <span className="text-sm text-purple-700 font-medium">
                Oviya is speaking
              </span>
            </>
          ) : (
            <>
              <Mic className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-500">
                Ready to listen
              </span>
            </>
          )}
        </div>
      </div>
      
      {/* Status Bar */}
      <div className="mt-3 h-1 bg-gray-200 rounded-full overflow-hidden">
        <motion.div
          className={`h-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}
          initial={{ width: 0 }}
          animate={{ width: isConnected ? '100%' : '0%' }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </motion.div>
  )
}


