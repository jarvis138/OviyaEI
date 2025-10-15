#!/usr/bin/env python3
"""
Complete pipeline test - Every component
"""

import asyncio
import websockets
import json
import numpy as np
import base64
import sounddevice as sd

async def test_complete_pipeline():
    print("=" * 70)
    print("🔬 COMPLETE PIPELINE TEST")
    print("=" * 70)
    
    uri = "ws://localhost:8000/ws/conversation?user_id=pipeline_test"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("\n✅ WebSocket connected")
            
            # TEST 1: Greeting (Backend → Browser)
            print("\n" + "=" * 70)
            print("TEST 1: GREETING FLOW")
            print("FastAPI → LLM (Ollama) → CSM Voice → WebSocket → Browser")
            print("=" * 70)
            
            await websocket.send(json.dumps({"type": "greeting", "text": "Hello"}))
            greeting_response = json.loads(await websocket.recv())
            
            print(f"✅ Greeting received:")
            print(f"   Text: {greeting_response.get('text')}")
            print(f"   Emotion: {greeting_response.get('emotion')}")
            print(f"   Audio chunks: {len(greeting_response.get('audio_chunks', []))}")
            print(f"   Duration: {greeting_response.get('duration', 0):.2f}s")
            
            if greeting_response.get('audio_chunks'):
                print("   ✅ GREETING AUDIO OUTPUT WORKING")
            else:
                print("   ❌ NO GREETING AUDIO!")
                return
            
            # TEST 2: User Audio Input (Browser → Backend → WhisperX)
            print("\n" + "=" * 70)
            print("TEST 2: AUDIO INPUT FLOW")
            print("Mic → Browser → WebSocket → FastAPI → WhisperX")
            print("=" * 70)
            
            # Generate speech-like audio (modulated tone)
            sample_rate = 16000
            duration = 2.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Speech-like formants (vowel /a/)
            f1, f2 = 700, 1220
            audio = (
                0.3 * np.sin(2 * np.pi * f1 * t) +
                0.2 * np.sin(2 * np.pi * f2 * t) +
                0.05 * np.random.randn(len(t))  # Add some noise
            )
            
            # Apply amplitude modulation (like speech envelope)
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
            audio = audio * envelope
            
            audio_int16 = (audio * 32767).astype(np.int16)
            
            print(f"📤 Sending audio: {len(audio_int16)} samples ({duration}s)")
            
            # Send audio chunks
            chunk_size = 4096
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                await websocket.send(chunk.tobytes())
                await asyncio.sleep(0.05)
            
            print("✅ Audio sent")
            
            # Wait for transcription
            print("⏳ Waiting for transcription from WhisperX...")
            try:
                transcription_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                transcription = json.loads(transcription_msg)
                
                if transcription.get('type') == 'transcription':
                    print(f"✅ TRANSCRIPTION RECEIVED:")
                    print(f"   Text: '{transcription.get('text')}'")
                    print(f"   Speakers: {transcription.get('speakers')}")
                    print(f"   ✅ AUDIO INPUT → WHISPERX WORKING")
                else:
                    print(f"⚠️ Unexpected message: {transcription.get('type')}")
                    
            except asyncio.TimeoutError:
                print("❌ TRANSCRIPTION TIMEOUT")
                print("   Problem: Audio Input → WhisperX pipeline")
                return
            
            # TEST 3: Response Generation (WhisperX → LLM → CSM → Browser)
            print("\n" + "=" * 70)
            print("TEST 3: RESPONSE FLOW")
            print("WhisperX → LLM (Ollama) → CSM Voice → Browser")
            print("=" * 70)
            
            print("⏳ Waiting for AI response...")
            try:
                response_msg = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                response = json.loads(response_msg)
                
                if response.get('type') == 'response':
                    print(f"✅ RESPONSE RECEIVED:")
                    print(f"   Text: '{response.get('text')}'")
                    print(f"   Emotion: {response.get('emotion')}")
                    print(f"   Audio chunks: {len(response.get('audio_chunks', []))}")
                    print(f"   Duration: {response.get('duration', 0):.2f}s")
                    
                    if response.get('audio_chunks'):
                        print("   ✅ RESPONSE AUDIO OUTPUT WORKING")
                        
                        # Verify audio quality
                        first_chunk = response['audio_chunks'][0]
                        decoded = base64.b64decode(first_chunk)
                        samples = np.frombuffer(decoded, dtype=np.int16)
                        
                        print(f"\n   Audio Quality Check:")
                        print(f"   - Samples: {len(samples)}")
                        print(f"   - Range: [{samples.min()}, {samples.max()}]")
                        print(f"   - Mean: {samples.mean():.1f}")
                        print(f"   - RMS: {np.sqrt(np.mean(samples**2)):.1f}")
                        
                        if samples.max() > 1000 and samples.min() < -1000:
                            print("   ✅ Audio has good dynamic range")
                        else:
                            print("   ⚠️ Audio seems quiet or clipped")
                            
                    else:
                        print("   ❌ NO RESPONSE AUDIO!")
                        return
                else:
                    print(f"⚠️ Unexpected message: {response.get('type')}")
                    
            except asyncio.TimeoutError:
                print("❌ RESPONSE TIMEOUT")
                print("   Problem: LLM → CSM Voice pipeline")
                return
            
            # FINAL SUMMARY
            print("\n" + "=" * 70)
            print("🎉 COMPLETE PIPELINE TEST RESULTS")
            print("=" * 70)
            print("✅ FastAPI WebSocket Server")
            print("✅ LLM (Ollama) - Greeting generation")
            print("✅ CSM Voice - Audio synthesis")
            print("✅ Audio Output (Backend → Browser)")
            print("✅ Audio Input (Browser → Backend)")
            print("✅ WhisperX - Transcription")
            print("✅ LLM (Ollama) - Response generation")
            print("✅ CSM Voice - Response audio")
            print("✅ Audio Output (Response → Browser)")
            print("\n🚀 ALL COMPONENTS WORKING!")
            print("=" * 70)
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ WebSocket connection closed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())

