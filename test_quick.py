#!/usr/bin/env python3
"""Quick test - just verify imports and basic functionality"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing imports...")

try:
    from reasoning.local_image_gen import get_image_generator
    print("✅ local_image_gen imported")
    
    from ui.dashboard_pro import create_dashboard
    print("✅ dashboard_pro imported")
    
    from pipeline.video_pipeline import VideoPipeline
    print("✅ video_pipeline imported")
    
    from memory.memory_gate import MemoryGate
    print("✅ memory_gate imported")
    
    print("\n🎯 Testing image generator initialization...")
    gen = get_image_generator()
    print(f"✅ Generator initialized (fallback={gen._use_fallback})")
    
    print("\n🎯 Testing pipeline initialization...")
    pipeline = VideoPipeline(video_source="test", use_llm=False)
    print("✅ Pipeline initialized")
    
    print("\n🎯 Testing memory gate...")
    gate = MemoryGate(base_threshold=0.3)
    gate.should_store(0.5)
    gate.should_store(0.8)
    adaptive = gate.get_adaptive_threshold()
    print(f"✅ Memory gate working (adaptive threshold={adaptive:.3f})")
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n📝 To test image generation:")
    print("   python -c \"from reasoning.local_image_gen import get_image_generator; img = get_image_generator().generate_image('test'); print('Generated:', img.size)\"")
    
    print("\n🚀 To launch dashboard:")
    print("   python -m ui.dashboard_pro")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
