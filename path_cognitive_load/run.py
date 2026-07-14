#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.main import main

if __name__ == "__main__":
    print("=" * 60)
    print("🧠  COGNITIVE LOAD DETECTION SYSTEM  v3.0")
    print("=" * 60)
    print("\n✓ Camera stays on throughout")
    print("✓ Live blink / cursor / face metrics")
    print("✓ Mixed difficulty (easy/medium/hard)")
    print("✓ Full analysis after 12 questions")
    print("✓ ADHD & Dyslexia risk assessment")
    print("\n🚀 Starting…\n")
    main()