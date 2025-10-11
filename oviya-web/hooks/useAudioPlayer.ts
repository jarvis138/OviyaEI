import React, { useState, useRef, useCallback } from 'react'

interface AudioPlayerState {
  isPlaying: boolean
  playAudio: (audioData: Uint8Array) => Promise<void>
  stopAudio: () => void
  currentTime: number
  duration: number
}

export const useAudioPlayer = (): AudioPlayerState => {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<AudioBufferSourceNode | null>(null)
  
  const playAudio = useCallback(async (audioData: Uint8Array) => {
    try {
      // Stop any currently playing audio
      stopAudio()
      
      // Create AudioContext if not exists
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }
      
      const audioContext = audioContextRef.current
      
      // Resume context if suspended
      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }
      
      // Convert Uint8Array to AudioBuffer
      const audioBuffer = await audioContext.decodeAudioData(audioData.buffer)
      
      // Create source node
      const source = audioContext.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContext.destination)
      
      sourceRef.current = source
      setDuration(audioBuffer.duration)
      
      // Handle playback events
      source.onended = () => {
        setIsPlaying(false)
        setCurrentTime(0)
        sourceRef.current = null
      }
      
      // Start playback
      source.start(0)
      setIsPlaying(true)
      
      // Update current time
      const updateTime = () => {
        if (isPlaying && audioContext.currentTime) {
          setCurrentTime(audioContext.currentTime)
          requestAnimationFrame(updateTime)
        }
      }
      updateTime()
      
    } catch (error) {
      console.error('Error playing audio:', error)
      setIsPlaying(false)
    }
  }, [isPlaying])
  
  const stopAudio = useCallback(() => {
    if (sourceRef.current) {
      try {
        sourceRef.current.stop()
      } catch (error) {
        // Source might already be stopped
      }
      sourceRef.current = null
    }
    
    setIsPlaying(false)
    setCurrentTime(0)
  }, [])
  
  return {
    isPlaying,
    playAudio,
    stopAudio,
    currentTime,
    duration
  }
}


