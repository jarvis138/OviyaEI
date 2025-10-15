import { NextPage } from 'next'
import Head from 'next/head'
import { LiveAIDemo } from '@/components/LiveAIDemo'
import { SimpleNav } from '@/components/SimpleNav'
import { CleanFooter } from '@/components/CleanFooter'

const Demo: NextPage = () => {
  return (
    <>
      <Head>
        <title>Oviya AI Demo - Live Voice Conversation</title>
        <meta name="description" content="Experience a full-screen demo of Oviya AI's voice conversation capabilities" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
        <SimpleNav />
        
        <main className="pt-20">
          <LiveAIDemo />
        </main>
        
        <CleanFooter />
      </div>
    </>
  )
}

export default Demo
