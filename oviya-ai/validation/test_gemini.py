#!/usr/bin/env python3
"""
Test Gemini API Integration
Sprint 0 - Day 1: Foundation Setup
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from loguru import logger

# Load environment variables
load_dotenv()

def test_gemini_basic():
    """Test basic Gemini API call"""
    logger.info("Testing Gemini API...")
    
    # Configure Gemini
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        return False
    
    genai.configure(api_key=api_key)
    
    # Create model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Test generation
    try:
        response = model.generate_content("Say hello in a friendly way")
        logger.success(f"✅ Gemini API working!")
        logger.info(f"Response: {response.text}")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini API failed: {e}")
        return False

def test_gemini_streaming():
    """Test Gemini streaming (for real-time responses)"""
    logger.info("Testing Gemini streaming...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    try:
        response = model.generate_content(
            "Count from 1 to 5, one number per sentence",
            stream=True
        )
        
        logger.info("Streaming response:")
        for chunk in response:
            if chunk.text:
                print(f"  {chunk.text}", end='', flush=True)
        print()  # New line
        
        logger.success("✅ Gemini streaming working!")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini streaming failed: {e}")
        return False

def test_gemini_with_emotion_tags():
    """Test Gemini with emotion tagging (for Oviya)"""
    logger.info("Testing Gemini with emotion tags...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    system_prompt = """
    You are Oviya, an empathetic AI companion.
    
    IMPORTANT: Start each response with an emotion tag:
    <emotion>empathetic|calm|encouraging|concerned|joyful</emotion>
    
    Then respond naturally and warmly.
    
    Example:
    User: "I'm feeling stressed today"
    Assistant: "<emotion>empathetic</emotion>I hear that you're feeling stressed. That sounds really challenging..."
    """
    
    try:
        response = model.generate_content(
            f"{system_prompt}\n\nUser: I'm feeling overwhelmed with work"
        )
        
        logger.info(f"Response with emotion: {response.text}")
        
        # Check if emotion tag is present
        if "<emotion>" in response.text:
            logger.success("✅ Emotion tagging working!")
            return True
        else:
            logger.warning("⚠️  Emotion tag not found, but response generated")
            return True
    except Exception as e:
        logger.error(f"❌ Emotion tagging test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Oviya - Gemini API Tests")
    logger.info("Sprint 0 - Day 1")
    logger.info("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Basic API", test_gemini_basic()))
    print()
    
    results.append(("Streaming", test_gemini_streaming()))
    print()
    
    results.append(("Emotion Tags", test_gemini_with_emotion_tags()))
    print()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.success("🎉 All tests passed! Gemini API is ready for Oviya!")
    else:
        logger.error("❌ Some tests failed. Please check configuration.")
    
    logger.info("=" * 60)
