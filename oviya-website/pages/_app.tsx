import type { AppProps } from 'next/app'
import { Toaster } from 'react-hot-toast'
import { useEffect } from 'react'
import '@/styles/globals.css'

export default function App({ Component, pageProps }: AppProps) {
  // Suppress MetaMask and other extension errors
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      // Suppress MetaMask and crypto wallet extension errors
      if (
        event.message?.includes('MetaMask') ||
        event.message?.includes('ethereum') ||
        event.filename?.includes('chrome-extension') ||
        event.filename?.includes('moz-extension')
      ) {
        event.preventDefault()
        event.stopPropagation()
        return false
      }
    }

    const handleRejection = (event: PromiseRejectionEvent) => {
      // Suppress extension-related promise rejections
      const reason = event.reason?.toString() || ''
      if (
        reason.includes('MetaMask') ||
        reason.includes('ethereum') ||
        reason.includes('chrome-extension')
      ) {
        event.preventDefault()
        return false
      }
    }

    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleRejection)

    return () => {
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleRejection)
    }
  }, [])

  return (
    <>
      <Component {...pageProps} />
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 4000,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </>
  )
}