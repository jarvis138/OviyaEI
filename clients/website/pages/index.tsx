import type { NextPage } from 'next'
import Head from 'next/head'
import { VoiceMode } from '@/components/VoiceMode'

const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>Oviya Voice - AI Conversation</title>
        <meta name="description" content="Experience natural voice conversations with Oviya AI" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet" />
      </Head>
      
      <VoiceMode />
    </>
  )
}

export default Home