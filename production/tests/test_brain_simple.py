#!/usr/bin/env python3
"""
Simple Brain Test
"""

import sys
from pathlib import Path

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from production.brain.llm_brain import OviyaBrain

def test_brain():
    print("Testing Oviya Brain...")
    
    brain = OviyaBrain(ollama_url="https://8f4e6bc7f36b4e.lhr.life/api/generate")
    
    print("\nTesting simple message...")
    response = brain.think("Hello, how are you?", "neutral")
    
    print(f"Response: {response}")
    print(f"Text: {response.get('text', 'No text')}")
    print(f"Emotion: {response.get('emotion', 'No emotion')}")
    print(f"Prosodic: {response.get('prosodic_text', 'No prosodic')}")

if __name__ == "__main__":
    test_brain()

