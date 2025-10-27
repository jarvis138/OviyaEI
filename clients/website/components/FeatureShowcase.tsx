import React from 'react'
import { motion } from 'framer-motion'
import { Mic, Brain, Heart, Users, BarChart3, Zap } from 'lucide-react'

const features = [
  {
    icon: Mic,
    title: 'Real-Time Voice',
    description: 'Natural conversation with instant responses and low latency',
    color: 'text-blue-600',
    bgColor: 'bg-blue-100'
  },
  {
    icon: Heart,
    title: '49-Emotion Detection',
    description: 'Advanced emotional intelligence that understands nuanced feelings',
    color: 'text-red-600',
    bgColor: 'bg-red-100'
  },
  {
    icon: Brain,
    title: 'Persistent Memory',
    description: 'Remembers you across conversations and builds lasting relationships',
    color: 'text-purple-600',
    bgColor: 'bg-purple-100'
  },
  {
    icon: Users,
    title: 'Multi-Speaker Support',
    description: 'Handles group conversations with speaker diarization',
    color: 'text-green-600',
    bgColor: 'bg-green-100'
  },
  {
    icon: BarChart3,
    title: 'Analytics Dashboard',
    description: 'Insights into conversation patterns and emotional trends',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-100'
  },
  {
    icon: Zap,
    title: 'Production Ready',
    description: 'Docker deployment with monitoring and A/B testing',
    color: 'text-yellow-600',
    bgColor: 'bg-yellow-100'
  }
]

export const FeatureShowcase: React.FC = () => {
  return (
    <section className="py-20 bg-gray-50">
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
            Advanced AI Capabilities
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Oviya combines cutting-edge AI with emotional intelligence for truly human-like interactions.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-white rounded-2xl p-8 shadow-sm hover:shadow-lg transition-shadow duration-300"
            >
              <div className={`w-12 h-12 ${feature.bgColor} rounded-xl flex items-center justify-center mb-6`}>
                <feature.icon className={`w-6 h-6 ${feature.color}`} />
              </div>
              
              <h3 className="text-xl font-medium text-gray-900 mb-3">
                {feature.title}
              </h3>
              
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          viewport={{ once: true }}
          className="text-center mt-16"
        >
          <p className="text-lg text-gray-600 mb-6">
            Ready to experience the future of AI conversation?
          </p>
          <button 
            onClick={() => document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' })}
            className="bg-purple-600 text-white px-8 py-4 rounded-full text-lg font-medium hover:bg-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
          >
            Try Oviya Now
          </button>
        </motion.div>
      </div>
    </section>
  )
}
