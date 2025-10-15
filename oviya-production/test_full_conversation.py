#!/usr/bin/env python3
"""
Test full conversation flow with audio input and output
"""

import asyncio
import websockets
import json
import numpy as np
import base64

async def test_full_conversation():
    uri = "ws://localhost:8000/ws/conversation?user_id=test_conversation"
    print("🔌 Connecting to WebSocket...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Step 1: Test greeting
            print("\n" + "="*60)
            print("STEP 1: Testing Greeting")
            print("="*60)
            
            greeting_msg = {"type": "greeting", "text": "Hello"}
            print(f"📤 Sending: {greeting_msg}")
            await websocket.send(json.dumps(greeting_msg))
            
            print("⏳ Waiting for greeting response...")
            greeting_response_str = await websocket.recv()
            greeting_response = json.loads(greeting_response_str)
            
            print(f"✅ Greeting received:")
            print(f"   Text: {greeting_response.get('text')}")
            print(f"   Emotion: {greeting_response.get('emotion')}")
            print(f"   Audio chunks: {len(greeting_response.get('audio_chunks', []))}")
            print(f"   Duration: {greeting_response.get('duration', 0):.2f}s")
            
            if greeting_response.get('audio_chunks'):
                print("   ✅ Greeting audio present")
            else:
                print("   ❌ No greeting audio!")
            
            # Step 2: Simulate user speaking (send audio)
            print("\n" + "="*60)
            print("STEP 2: Testing User Audio Input")
            print("="*60)
            
            # Generate 3 seconds of test audio (16kHz, mono, PCM)
            sample_rate = 16000
            duration = 3.0
            num_samples = int(sample_rate * duration)
            
            # Create audio with some pattern (not just random noise)
            t = np.linspace(0, duration, num_samples)
            # Mix of frequencies to simulate speech
            audio = (
                0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
                0.3 * np.sin(2 * np.pi * 400 * t) +  # Mid frequency
                0.2 * np.sin(2 * np.pi * 800 * t)    # High frequency
            )
            
            # Convert to int16 PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            print(f"📤 Sending audio data: {len(audio_int16)} samples ({duration}s)")
            
            # Send audio in chunks (like real microphone would)
            chunk_size = 4096
            chunks_sent = 0
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                await websocket.send(chunk.tobytes())
                chunks_sent += 1
                
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.05)
            
            print(f"✅ Sent {chunks_sent} audio chunks")
            
            # Step 3: Wait for transcription
            print("\n⏳ Waiting for transcription...")
            try:
                transcription_str = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                transcription = json.loads(transcription_str)
                
                if transcription.get('type') == 'transcription':
                    print(f"✅ Transcription received:")
                    print(f"   Text: '{transcription.get('text')}'")
                    print(f"   Speakers: {transcription.get('speakers', [])}")
                    print(f"   Word timestamps: {len(transcription.get('word_timestamps', []))}")
                    
                    # Step 4: Wait for AI response with audio
                    print("\n⏳ Waiting for AI response...")
                    response_str = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    response = json.loads(response_str)
                    
                    if response.get('type') == 'response':
                        print(f"✅ AI Response received:")
                        print(f"   Text: '{response.get('text')}'")
                        print(f"   Emotion: {response.get('emotion')}")
                        print(f"   Audio chunks: {len(response.get('audio_chunks', []))}")
                        print(f"   Duration: {response.get('duration', 0):.2f}s")
                        
                        if response.get('audio_chunks'):
                            print("   ✅ Response audio present")
                            
                            # Verify audio data
                            first_chunk = response['audio_chunks'][0]
                            print(f"   First chunk size: {len(first_chunk)} bytes (base64)")
                            
                            # Decode to verify it's valid
                            try:
                                decoded = base64.b64decode(first_chunk)
                                decoded_int16 = np.frombuffer(decoded, dtype=np.int16)
                                print(f"   Decoded samples: {len(decoded_int16)}")
                                print(f"   Sample range: [{decoded_int16.min()}, {decoded_int16.max()}]")
                                print("   ✅ Audio data is valid PCM")
                            except Exception as e:
                                print(f"   ❌ Audio decode error: {e}")
                        else:
                            print("   ❌ No response audio!")
                    else:
                        print(f"⚠️ Unexpected message type: {response.get('type')}")
                else:
                    print(f"⚠️ Expected transcription, got: {transcription.get('type')}")
                    
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for transcription/response")
                print("   This might mean:")
                print("   - Remote WhisperX is not responding")
                print("   - Audio buffer not full enough (< 1 second)")
                print("   - Backend processing error")
            
            print("\n" + "="*60)
            print("🏁 CONVERSATION TEST COMPLETE")
            print("="*60)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_conversation())

