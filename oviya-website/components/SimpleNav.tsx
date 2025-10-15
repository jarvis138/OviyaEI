import React from 'react'
import Link from 'next/link'
import { Heart, Github, Twitter, Mail } from 'lucide-react'

export const SimpleNav: React.FC = () => {
  return (
    <nav className="fixed top-0 w-full bg-white/80 backdrop-blur-sm z-50 border-b border-gray-100">
      <div className="max-w-6xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-medium text-gray-900">Oviya</span>
          </Link>
          
          {/* Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <a href="#demo" className="text-gray-600 hover:text-purple-600 transition-colors">
              Demo
            </a>
            <a href="#features" className="text-gray-600 hover:text-purple-600 transition-colors">
              Features
            </a>
            <a href="#about" className="text-gray-600 hover:text-purple-600 transition-colors">
              About
            </a>
          </div>
          
          {/* Social Links */}
          <div className="flex items-center space-x-4">
            <a
              href="https://github.com/oviya-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="https://twitter.com/oviya_ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Twitter className="w-5 h-5" />
            </a>
            <a
              href="mailto:hello@oviya.ai"
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>
      </div>
    </nav>
  )
}
