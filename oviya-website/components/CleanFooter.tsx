import React from 'react'
import { Heart, Github, Twitter, Mail } from 'lucide-react'

export const CleanFooter: React.FC = () => {
  return (
    <footer className="bg-white border-t border-gray-100 py-12">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between">
          {/* Logo */}
          <div className="flex items-center space-x-2 mb-6 md:mb-0">
            <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center">
              <Heart className="w-4 h-4 text-white" />
            </div>
            <span className="text-lg font-medium text-gray-900">Oviya</span>
          </div>
          
          {/* Links */}
          <div className="flex items-center space-x-8 mb-6 md:mb-0">
            <a href="#demo" className="text-gray-600 hover:text-purple-600 transition-colors text-sm">
              Demo
            </a>
            <a href="#features" className="text-gray-600 hover:text-purple-600 transition-colors text-sm">
              Features
            </a>
            <a href="#about" className="text-gray-600 hover:text-purple-600 transition-colors text-sm">
              About
            </a>
            <a href="/privacy" className="text-gray-600 hover:text-purple-600 transition-colors text-sm">
              Privacy
            </a>
            <a href="/terms" className="text-gray-600 hover:text-purple-600 transition-colors text-sm">
              Terms
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
        
        {/* Copyright */}
        <div className="mt-8 pt-8 border-t border-gray-100 text-center">
          <p className="text-sm text-gray-500">
            © 2024 Oviya AI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
