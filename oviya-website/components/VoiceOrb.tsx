import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Mic, Volume2 } from 'lucide-react'

interface VoiceOrbProps {
  isListening: boolean
  isSpeaking: boolean
  isConnected: boolean
  audioLevel?: number
  onClick: () => void
}

export const VoiceOrb: React.FC<VoiceOrbProps> = ({
  isListening,
  isSpeaking,
  isConnected,
  audioLevel = 0,
  onClick
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  // Animated waveform visualization
  useEffect(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    let animationId: number
    let phase = 0
    
    const draw = () => {
      const width = canvas.width
      const height = canvas.height
      const centerX = width / 2
      const centerY = height / 2
      
      ctx.clearRect(0, 0, width, height)
      
      if (isListening || isSpeaking) {
        // Draw animated waveform rings with audio level
        const rings = 3
        const levelMultiplier = isListening ? (1 + audioLevel * 0.5) : 1
        for (let i = 0; i < rings; i++) {
          const radius = (120 + i * 40 + Math.sin(phase + i * 0.5) * 10) * levelMultiplier
          const opacity = (0.3 - (i * 0.1)) * (0.5 + audioLevel * 0.5)
          
          ctx.beginPath()
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
          ctx.strokeStyle = isSpeaking 
            ? `rgba(139, 92, 246, ${opacity})` 
            : `rgba(59, 130, 246, ${opacity})`
          ctx.lineWidth = 2
          ctx.stroke()
        }
        
        phase += 0.05
      }
      
      animationId = requestAnimationFrame(draw)
    }
    
    draw()
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId)
      }
    }
  }, [isListening, isSpeaking])
  
  return (
    <div className="relative w-80 h-80 flex items-center justify-center">
      {/* Canvas for waveform */}
      <canvas
        ref={canvasRef}
        width={320}
        height={320}
        className="absolute inset-0"
      />
      
      {/* Outer glow rings */}
      {(isListening || isSpeaking) && (
        <>
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{
              background: isSpeaking 
                ? 'radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%)'
                : 'radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, transparent 70%)'
            }}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.5, 0.8, 0.5]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{
              background: isSpeaking 
                ? 'radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%)'
                : 'radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%)'
            }}
            animate={{
              scale: [1, 1.4, 1],
              opacity: [0.3, 0.6, 0.3]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5
            }}
          />
        </>
      )}
      
      {/* Main orb button */}
      <motion.button
        onClick={onClick}
        disabled={!isConnected}
        className={`relative z-10 w-48 h-48 rounded-full flex items-center justify-center
                   shadow-2xl cursor-pointer transition-all duration-300
                   ${!isConnected ? 'opacity-50 cursor-not-allowed' : ''}
                   ${isListening ? 'bg-gradient-to-br from-blue-500 to-blue-700' : 
                     isSpeaking ? 'bg-gradient-to-br from-purple-500 to-purple-700' : 
                     'bg-gradient-to-br from-purple-600 to-blue-600'}`}
        whileHover={isConnected ? { scale: 1.05 } : {}}
        whileTap={isConnected ? { scale: 0.95 } : {}}
        animate={isListening ? {
          scale: [1, 1.05, 1],
        } : isSpeaking ? {
          scale: [1, 1.08, 1],
        } : {}}
        transition={{
          duration: 1.5,
          repeat: (isListening || isSpeaking) ? Infinity : 0,
          ease: "easeInOut"
        }}
      >
        {/* Inner glow */}
        <motion.div
          className="absolute inset-8 rounded-full bg-white/20"
          animate={isListening || isSpeaking ? {
            opacity: [0.2, 0.4, 0.2],
            scale: [0.9, 1, 0.9]
          } : {}}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        
        {/* Icon */}
        {isSpeaking ? (
          <Volume2 className="w-20 h-20 text-white z-10" />
        ) : (
          <Mic className="w-20 h-20 text-white z-10" />
        )}
      </motion.button>
      
      {/* Connection status indicator */}
      {!isConnected && (
        <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2">
          <div className="flex items-center gap-2 bg-red-500/20 backdrop-blur-sm rounded-full px-4 py-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-red-300 text-sm">Connecting...</span>
          </div>
        </div>
      )}
    </div>
  )
}
