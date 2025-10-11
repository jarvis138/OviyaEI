import React from 'react'
import { motion } from 'framer-motion'
import { Heart, Zap, Moon, Smile, Frown } from 'lucide-react'

interface EmotionSelectorProps {
  currentEmotion: string
  onEmotionChange: (emotion: string) => void
  disabled?: boolean
}

const emotions = [
  {
    id: 'empathetic',
    name: 'Empathetic',
    icon: Heart,
    color: 'text-pink-500',
    bgColor: 'bg-pink-100',
    description: 'Warm and understanding'
  },
  {
    id: 'encouraging',
    name: 'Encouraging',
    icon: Zap,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-100',
    description: 'Positive and uplifting'
  },
  {
    id: 'calm',
    name: 'Calm',
    icon: Moon,
    color: 'text-blue-500',
    bgColor: 'bg-blue-100',
    description: 'Peaceful and soothing'
  },
  {
    id: 'joyful',
    name: 'Joyful',
    icon: Smile,
    color: 'text-green-500',
    bgColor: 'bg-green-100',
    description: 'Happy and cheerful'
  },
  {
    id: 'concerned',
    name: 'Concerned',
    icon: Frown,
    color: 'text-orange-500',
    bgColor: 'bg-orange-100',
    description: 'Caring and supportive'
  }
]

export const EmotionSelector: React.FC<EmotionSelectorProps> = ({
  currentEmotion,
  onEmotionChange,
  disabled = false
}) => {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 text-center">
        Choose Oviya's Emotion
      </h3>
      
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {emotions.map((emotion) => {
          const Icon = emotion.icon
          const isSelected = currentEmotion === emotion.id
          
          return (
            <motion.button
              key={emotion.id}
              className={`
                p-4 rounded-xl border-2 transition-all duration-200
                ${isSelected 
                  ? `${emotion.bgColor} border-current ${emotion.color}` 
                  : 'bg-white border-gray-200 hover:border-gray-300'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
              onClick={() => !disabled && onEmotionChange(emotion.id)}
              disabled={disabled}
              whileHover={!disabled ? { scale: 1.02 } : {}}
              whileTap={!disabled ? { scale: 0.98 } : {}}
            >
              <div className="flex flex-col items-center space-y-2">
                <Icon className={`w-6 h-6 ${isSelected ? emotion.color : 'text-gray-400'}`} />
                <div className="text-center">
                  <div className={`text-sm font-medium ${isSelected ? emotion.color : 'text-gray-700'}`}>
                    {emotion.name}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {emotion.description}
                  </div>
                </div>
              </div>
            </motion.button>
          )
        })}
      </div>
      
      {/* Current Selection Indicator */}
      <div className="text-center">
        <motion.div
          key={currentEmotion}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-gray-100 rounded-full"
        >
          <span className="text-sm text-gray-600">
            Current: <span className="font-medium text-gray-900">
              {emotions.find(e => e.id === currentEmotion)?.name}
            </span>
          </span>
        </motion.div>
      </div>
    </div>
  )
}


