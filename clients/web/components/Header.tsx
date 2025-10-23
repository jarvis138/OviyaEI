import React from 'react'
import Link from 'next/link'
import { Heart, Github, Twitter, Mail } from 'lucide-react'

export const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">Oviya</span>
          </Link>
          
          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            <Link href="/" className="text-gray-600 hover:text-purple-600 transition-colors">
              Home
            </Link>
            <Link href="/about" className="text-gray-600 hover:text-purple-600 transition-colors">
              About
            </Link>
            <Link href="/privacy" className="text-gray-600 hover:text-purple-600 transition-colors">
              Privacy
            </Link>
            <Link href="/terms" className="text-gray-600 hover:text-purple-600 transition-colors">
              Terms
            </Link>
          </nav>
          
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
    </header>
  )
}


