#!/usr/bin/env python3
"""
Test Hugging Face Access to CSM Model
Sprint 0 - Day 2
"""

import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

def test_hf_token():
    """Test if Hugging Face token is configured"""
    logger.info("Testing Hugging Face token...")
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not token:
        logger.error("❌ HUGGINGFACE_TOKEN not found in .env")
        logger.info("   Add it with: echo 'HUGGINGFACE_TOKEN=hf_...' >> .env")
        return False
    
    if not token.startswith('hf_'):
        logger.error("❌ Token doesn't start with 'hf_'")
        logger.info("   Make sure you copied the full token")
        return False
    
    logger.success(f"✅ Token found: {token[:10]}...{token[-5:]}")
    return True

def test_hf_cli():
    """Test if huggingface_hub is installed"""
    logger.info("Testing huggingface_hub installation...")
    
    try:
        from huggingface_hub import HfApi
        logger.success("✅ huggingface_hub installed")
        return True
    except ImportError:
        logger.error("❌ huggingface_hub not installed")
        logger.info("   Install with: pip install huggingface_hub")
        return False

def test_csm_access():
    """Test if we can access CSM model"""
    logger.info("Testing CSM model access...")
    
    try:
        from huggingface_hub import HfApi
        
        token = os.getenv('HUGGINGFACE_TOKEN')
        api = HfApi()
        
        # Try to get model info
        model_id = "sesame-ai/csm-1b"
        logger.info(f"Checking access to {model_id}...")
        
        try:
            model_info = api.model_info(model_id, token=token)
            logger.success(f"✅ Access granted to {model_id}")
            logger.info(f"   Model ID: {model_info.id}")
            logger.info(f"   Downloads: {model_info.downloads}")
            return True
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                logger.error("❌ Access denied to CSM model")
                logger.info("   Go to: https://huggingface.co/sesame-ai/csm-1b")
                logger.info("   Click 'Request Access' and wait for approval")
            else:
                logger.error(f"❌ Error accessing model: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Hugging Face Access Test")
    logger.info("Sprint 0 - Day 2")
    logger.info("=" * 60)
    logger.info("")
    
    results = []
    
    # Test 1: Token configured
    results.append(("Token Configured", test_hf_token()))
    print()
    
    # Test 2: HF Hub installed
    results.append(("HF Hub Installed", test_hf_cli()))
    print()
    
    # Test 3: CSM access
    if results[0][1] and results[1][1]:  # Only if previous tests passed
        results.append(("CSM Access", test_csm_access()))
    else:
        logger.warning("⚠️  Skipping CSM access test (prerequisites failed)")
        results.append(("CSM Access", False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.success("\n🎉 All tests passed! Ready to install CSM!")
        logger.info("\nNext steps:")
        logger.info("  1. Choose setup: RunPod (recommended) or Local GPU")
        logger.info("  2. Follow: validation/csm-benchmark/SETUP.md")
        logger.info("  3. Run: python basic_test.py")
    else:
        logger.error("\n❌ Some tests failed. Please fix issues above.")
    
    logger.info("=" * 60)

