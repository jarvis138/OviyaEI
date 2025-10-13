#!/usr/bin/env python3
"""
Debug LLM Connection
"""

import requests
import json

def test_llm_connection():
    ollama_url = "https://8f4e6bc7f36b4e.lhr.life/api/generate"
    
    request_payload = {
        "model": "qwen2.5:7b",
        "prompt": "Hello, test connection",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 50,
            "stop": ["User:", "\n\n"]
        }
    }
    
    print(f"Testing LLM connection to: {ollama_url}")
    print(f"Payload: {json.dumps(request_payload, indent=2)}")
    
    try:
        response = requests.post(
            ollama_url,
            json=request_payload,
            timeout=15
        )
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            print(f"✅ Success! LLM Response: {result.get('response', 'No response')}")
        else:
            print(f"❌ Error! Status: {response.status_code}")
            print(f"Response Text: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_llm_connection()

