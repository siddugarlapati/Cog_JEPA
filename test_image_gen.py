#!/usr/bin/env python3
"""
Quick test for image generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from reasoning.local_image_gen import get_image_generator

def test_generation():
    print("🎨 Testing image generation...")
    
    gen = get_image_generator()
    
    prompt = "A dramatic cinematic scene of a sniper in action, professional cinematography, 4k"
    print(f"\n📝 Prompt: {prompt}")
    print("⏳ Generating (this may take 10-30 seconds)...")
    
    img = gen.generate_image(prompt, size="512x512")
    
    if img:
        print(f"✅ Success! Image size: {img.size}")
        print(f"   Mode: {img.mode}")
        
        # Save test output
        output_path = "data/test_generation.png"
        os.makedirs("data", exist_ok=True)
        img.save(output_path)
        print(f"💾 Saved to: {output_path}")
        
        return True
    else:
        print("❌ Generation failed")
        return False

if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)
