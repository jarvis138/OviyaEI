#!/usr/bin/env python3
"""
Test Oviya Brain with Local Ollama
"""

import sys
from pathlib import Path
import requests

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from production.brain.llm_brain import OviyaBrain

def check_ollama():
    """Check if Ollama is running and has models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running!")
            print(f"📦 Available models: {[m['name'] for m in models]}")

            # Check for qwen2.5:7b
            has_qwen = any('qwen2.5' in m['name'] for m in models)
            if has_qwen:
                print("✅ Qwen2.5 model found!")
                return True
            else:
                print("⚠️  Qwen2.5 model not found. Using first available model.")
                return True
        else:
            print("❌ Ollama responded but with error")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running with: ollama serve")
        return False

def test_brain():
    """Test Oviya Brain with local Ollama"""
    print("🧠 Testing Oviya Brain with Local Ollama...")

    # Check Ollama first
    if not check_ollama():
        print("❌ Aborting test - Ollama not available")
        return

    try:
        # Use local Ollama with correct persona config path
        brain = OviyaBrain(
            persona_config_path="production/config/oviya_persona.json"
        )  # Uses http://localhost:11434/api/generate by default

        # Show persona info
        persona = brain.persona_config.get('persona', {})
        print(f"\n👤 Loaded Persona: {persona.get('name', 'Unknown')}")
        print(f"   Description: {persona.get('description', 'No description')}")
        print(f"   Traits: {persona.get('personality_traits', [])}")
        print(f"   System Prompt: {brain.system_prompt[:100]}...")
        print(f"   Model: {brain.model_name}")

        print("\n🗣️  Testing simple message...")
        response = brain.think("Hello, how are you?", "neutral")

        print("\n📝 Response Details:")
        print(f"   Text: {response.get('text', 'No text')}")
        print(f"   Emotion: {response.get('emotion', 'No emotion')}")
        print(f"   Intensity: {response.get('intensity', 'No intensity')}")
        print(f"   Prosodic: {response.get('prosodic_text', 'No prosodic')[:100]}...")

        if 'emotional_state' in response:
            state = response['emotional_state']
            print(f"   Energy: {state.get('energy_level', 'N/A')}")
            print(f"   Warmth: {state.get('warmth', 'N/A')}")

        # Test a more emotionally complex message (avoiding safety triggers)
        print("\n🗣️  Testing emotional message...")
        response2 = brain.think("I'm feeling really overwhelmed with all my tasks today", "overwhelmed")

        print("\n📝 Emotional Response Details:")
        print(f"   Text: {response2.get('text', 'No text')}")
        print(f"   Emotion: {response2.get('emotion', 'No emotion')}")
        print(f"   Intensity: {response2.get('intensity', 'No intensity')}")
        print(f"   Prosodic: {response2.get('prosodic_text', 'No prosodic')[:100]}...")

        if 'emotional_state' in response2:
            state = response2['emotional_state']
            print(f"   Energy: {state.get('energy_level', 'N/A')}")
            print(f"   Warmth: {state.get('warmth', 'N/A')}")

        print("\n✅ Brain test completed successfully!")
        print("💡 Oviya should respond with empathy, validation, and situational awareness")

    except Exception as e:
        print(f"❌ Brain test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_brain()
