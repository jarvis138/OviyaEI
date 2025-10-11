#!/usr/bin/env python3
"""
CSM Basic Test - Sprint 0 Day 2
Measures cold start, warm inference, and audio quality
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add CSM to path if running from csm directory
csm_path = Path(__file__).parent / "csm"
if csm_path.exists():
    sys.path.insert(0, str(csm_path))

try:
    from generator import load_csm_1b
    import torchaudio
    import torch
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Please install CSM first. See SETUP.md")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

def test_cold_start():
    """Test 1: Measure cold start time"""
    logger.info("=" * 60)
    logger.info("Test 1: Cold Start Latency")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cpu":
        logger.warning("⚠️  Running on CPU - will be SLOW!")
        logger.warning("   For accurate results, use GPU")
    
    # Measure cold start
    logger.info("Loading CSM model...")
    start_time = time.time()
    
    try:
        generator = load_csm_1b(device=device)
        cold_start_time = time.time() - start_time
        
        logger.success(f"✅ Model loaded in {cold_start_time:.2f}s")
        
        # Success criteria
        if cold_start_time < 30:
            logger.success(f"   Cold start: {cold_start_time:.2f}s < 30s ✅")
        else:
            logger.warning(f"   Cold start: {cold_start_time:.2f}s > 30s ⚠️")
        
        return generator, cold_start_time
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None, None

def test_warm_inference(generator):
    """Test 2: Measure warm inference time"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Warm Inference Latency")
    logger.info("=" * 60)
    
    test_texts = [
        "Hello from Sesame.",
        "This is a test of the CSM model.",
        "I understand how you're feeling."
    ]
    
    latencies = []
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\nGeneration {i}/3: \"{text}\"")
        
        start_time = time.time()
        try:
            audio = generator.generate(
                text=text,
                speaker=0,
                context=[],
                max_audio_length_ms=10_000
            )
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Save audio
            output_file = OUTPUT_DIR / f"test_{i}.wav"
            torchaudio.save(
                str(output_file),
                audio.unsqueeze(0).cpu(),
                generator.sample_rate
            )
            
            logger.success(f"✅ Generated in {latency:.2f}s")
            logger.info(f"   Saved to: {output_file}")
            
            # Check against target
            if latency < 2.0:
                logger.success(f"   Latency: {latency:.2f}s < 2.0s ✅")
            else:
                logger.warning(f"   Latency: {latency:.2f}s > 2.0s ⚠️")
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            latencies.append(None)
    
    # Calculate average
    valid_latencies = [l for l in latencies if l is not None]
    if valid_latencies:
        avg_latency = sum(valid_latencies) / len(valid_latencies)
        logger.info(f"\n📊 Average latency: {avg_latency:.2f}s")
        return avg_latency
    else:
        return None

def test_audio_quality(generator):
    """Test 3: Generate sample for quality check"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Audio Quality Check")
    logger.info("=" * 60)
    
    # Generate a longer, more natural sample
    text = "I understand how you're feeling. That sounds really challenging. I'm here to listen if you want to talk about it."
    
    logger.info(f"Generating: \"{text}\"")
    
    try:
        start_time = time.time()
        audio = generator.generate(
            text=text,
            speaker=0,
            context=[],
            max_audio_length_ms=20_000,
            temperature=0.7,
            do_sample=True
        )
        latency = time.time() - start_time
        
        # Save audio
        output_file = OUTPUT_DIR / "quality_test.wav"
        torchaudio.save(
            str(output_file),
            audio.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        logger.success(f"✅ Generated in {latency:.2f}s")
        logger.info(f"   Saved to: {output_file}")
        logger.info(f"   Sample rate: {generator.sample_rate}Hz")
        logger.info(f"   Duration: {len(audio) / generator.sample_rate:.2f}s")
        logger.info("\n   👂 Please listen to the audio file and rate quality:")
        logger.info("      1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, 5 = Excellent")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quality test failed: {e}")
        return False

def save_results(cold_start, avg_latency):
    """Save test results to file"""
    results_file = OUTPUT_DIR / "day_2_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("CSM Basic Test Results - Day 2\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Cold Start Time: {cold_start:.2f}s\n")
        f.write(f"Average Warm Latency: {avg_latency:.2f}s\n\n")
        f.write("Success Criteria:\n")
        f.write(f"  Cold start < 30s: {'✅ PASS' if cold_start < 30 else '❌ FAIL'}\n")
        f.write(f"  Warm latency < 2s: {'✅ PASS' if avg_latency < 2.0 else '❌ FAIL'}\n")
    
    logger.info(f"\n📄 Results saved to: {results_file}")

def main():
    """Run all CSM tests"""
    logger.info("=" * 60)
    logger.info("Oviya - CSM Basic Tests")
    logger.info("Sprint 0 - Day 2")
    logger.info("=" * 60)
    logger.info("")
    
    # Test 1: Cold start
    generator, cold_start = test_cold_start()
    if generator is None:
        logger.error("❌ Cannot proceed without model")
        return False
    
    # Test 2: Warm inference
    avg_latency = test_warm_inference(generator)
    if avg_latency is None:
        logger.error("❌ Warm inference tests failed")
        return False
    
    # Test 3: Audio quality
    quality_ok = test_audio_quality(generator)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Summary")
    logger.info("=" * 60)
    logger.info(f"Cold Start: {cold_start:.2f}s {'✅' if cold_start < 30 else '❌'}")
    logger.info(f"Avg Warm Latency: {avg_latency:.2f}s {'✅' if avg_latency < 2.0 else '❌'}")
    logger.info(f"Audio Quality: {'✅ Generated' if quality_ok else '❌ Failed'}")
    
    # Save results
    save_results(cold_start, avg_latency)
    
    # Decision
    logger.info("\n" + "=" * 60)
    if cold_start < 30 and avg_latency < 2.0 and quality_ok:
        logger.success("🎉 All tests PASSED!")
        logger.success("✅ Ready to proceed to Day 3: Audio Context Testing")
    else:
        logger.warning("⚠️  Some tests need attention")
        logger.info("   Review results and optimize if needed")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

