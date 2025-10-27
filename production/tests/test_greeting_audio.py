#!/usr/bin/env python3
"""
Test if greeting generates audio properly
"""

import asyncio
import websockets
import json

async def test_greeting_with_audio():
    uri = "ws://localhost:8000/ws/conversation?user_id=test_audio_user"
    print("🔌 Connecting to WebSocket...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Send greeting message
            greeting_message = {"type": "greeting", "text": "Hello"}
            print(f"\n📤 Sending: {greeting_message}")
            await websocket.send(json.dumps(greeting_message))
            
            print("⏳ Waiting for response...")
            response_str = await websocket.recv()
            response = json.loads(response_str)
            
            print("\n" + "="*60)
            print("📨 RESPONSE RECEIVED:")
            print("="*60)
            print(f"Type: {response.get('type')}")
            print(f"Text: {response.get('text')}")
            print(f"Emotion: {response.get('emotion')}")
            print(f"Duration: {response.get('duration', 0):.2f}s")
            
            audio_chunks = response.get('audio_chunks', [])
            print(f"\n🎵 Audio Chunks: {len(audio_chunks)}")
            
            if audio_chunks:
                print("✅ AUDIO IS BEING GENERATED!")
                print(f"   First chunk length: {len(audio_chunks[0])} bytes (base64)")
                print(f"   Last chunk length: {len(audio_chunks[-1])} bytes (base64)")
                
                # Calculate total audio size
                total_size = sum(len(chunk) for chunk in audio_chunks)
                print(f"   Total audio data: {total_size:,} bytes (base64)")
                print(f"   Estimated decoded size: ~{int(total_size * 0.75):,} bytes")
            else:
                print("❌ NO AUDIO CHUNKS!")
                
            print("="*60)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_greeting_with_audio())

