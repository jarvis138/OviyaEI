import { NextPage } from 'next'
import Head from 'next/head'
import { VoiceChat } from '@/components/VoiceChat'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'

const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>Oviya AI - Real-time Voice AI</title>
        <meta name="description" content="Experience empathetic AI conversations with real-time voice interaction" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50">
        <Header />
        
        <main className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            {/* Hero Section */}
            <div className="text-center mb-12">
              <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
                Meet <span className="text-purple-600">Oviya</span>
              </h1>
              <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                Your empathetic AI companion with real-time voice interaction. 
                Experience natural conversations powered by advanced speech AI.
              </p>
              
              <div className="flex flex-wrap justify-center gap-4 text-sm text-gray-500">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  Real-time Voice
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  Empathetic AI
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  Privacy First
                </div>
              </div>
            </div>
            
            {/* Voice Chat Component */}
            <VoiceChat />
          </div>
        </main>
        
        <Footer />
      </div>
    </>
  )
}

export default Home


