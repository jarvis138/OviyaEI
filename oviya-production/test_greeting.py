#!/usr/bin/env python3
"""
Test script to verify greeting functionality
"""
import asyncio
import websockets
import json

async def test_greeting():
    uri = "ws://localhost:8000/ws/conversation?user_id=test_user"
    
    print("🔌 Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connected!")
        
        # Send greeting request
        greeting_msg = json.dumps({"type": "greeting", "text": "Hello"})
        print(f"📤 Sending: {greeting_msg}")
        await websocket.send(greeting_msg)
        
        # Wait for response
        print("⏳ Waiting for response...")
        response = await websocket.recv()
        data = json.loads(response)
        
        print(f"\n📨 Received response:")
        print(f"   Type: {data.get('type')}")
        print(f"   Text: {data.get('text')}")
        print(f"   Emotion: {data.get('emotion')}")
        print(f"   Audio chunks: {len(data.get('audio_chunks', []))}")
        print(f"   Duration: {data.get('duration')}")
        
        if data.get('audio_chunks'):
            print(f"\n✅ Audio chunks received successfully!")
            print(f"   First chunk length: {len(data['audio_chunks'][0])}")
        else:
            print(f"\n❌ No audio chunks received!")

if __name__ == "__main__":
    asyncio.run(test_greeting())

