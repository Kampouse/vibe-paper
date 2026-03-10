#!/usr/bin/env python3
"""Test the consolidation-driven training system"""

import asyncio
import httpx
import time

MEMORY_URL = "http://localhost:8888"
RL_URL = "http://localhost:30000"

async def test_full_loop():
    """Test the complete loop: ingest → consolidate → train"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        print("=" * 60)
        print("TEST: Consolidation-Driven Training")
        print("=" * 60)
        
        # 1. Clear existing data
        print("\n1. Clearing existing data...")
        try:
            await client.post(f"{MEMORY_URL}/clear")
            await client.post(f"{RL_URL}/clear-samples?sample_type=all")
            print("   ✓ Cleared")
        except Exception as e:
            print(f"   ⚠ Could not clear: {e}")
        
        # 2. Ingest sample conversations
        print("\n2. Ingesting sample conversations...")
        
        conversations = [
            ("User asked for file contents. Agent read the file successfully.", "session/1"),
            ("User: 'That was too verbose, keep it shorter'", "feedback"),
            ("Agent provided a 3-line summary instead. User: 'Perfect!'", "session/1"),
            ("User asked to run tests. Agent ran them without checking if they exist first.", "session/2"),
            ("Tests failed: file not found. User: 'Check if files exist first next time'", "feedback"),
            ("User asked about preferences. Agent: 'You prefer short answers.'", "session/3"),
            ("User: 'Yes, I like concise responses without fluff'", "session/3"),
            ("Agent used wrong command syntax. User corrected: 'Use -la not -al'", "feedback"),
            ("Agent tried again with correct syntax. Success!", "session/4"),
            ("User: 'Good job remembering the correct syntax'", "feedback"),
        ]
        
        for i, (text, source) in enumerate(conversations):
            try:
                resp = await client.post(
                    f"{MEMORY_URL}/ingest",
                    json={"text": text, "source": source}
                )
                print(f"   [{i+1}] {text[:50]}...")
            except Exception as e:
                print(f"   ⚠ Failed: {e}")
        
        # 3. Check what we have
        print("\n3. Checking memory stats...")
        resp = await client.get(f"{MEMORY_URL}/stats")
        stats = resp.json()
        print(f"   Total memories: {stats.get('total_memories', 0)}")
        print(f"   Unconsolidated: {stats.get('unconsolidated', 0)}")
        
        # 4. Trigger consolidation
        print("\n4. Triggering consolidation...")
        resp = await client.post(f"{MEMORY_URL}/consolidate")
        result = resp.json()
        
        print(f"   Memories processed: {result.get('memories_processed', 0)}")
        print(f"   Insights created: {result.get('insights_created', 0)}")
        print(f"   Training samples generated: {result.get('training_samples', 0)}")
        print(f"   Samples sent to RL: {result.get('samples_sent', 0)}")
        
        # 5. Show detected patterns
        print("\n5. Detected patterns:")
        for i, p in enumerate(result.get('patterns', [])[:5]):
            print(f"   [{i+1}] [{p['type']}] {p['pattern'][:60]}...")
            print(f"       Reward: {p['reward']}")
        
        # 6. Check RL server received samples
        print("\n6. Checking RL server samples...")
        resp = await client.get(f"{RL_URL}/samples")
        samples = resp.json()
        print(f"   Feedback samples: {samples.get('feedback_samples', 0)}")
        print(f"   Consolidation samples: {samples.get('consolidation_samples', 0)}")
        print(f"   Total: {samples.get('total', 0)}")
        
        # 7. View learned patterns
        print("\n7. Learned patterns by type:")
        resp = await client.get(f"{RL_URL}/patterns")
        patterns = resp.json()
        
        for ptype, plist in patterns.get('patterns_by_type', {}).items():
            print(f"\n   {ptype}:")
            for p in plist[:3]:
                print(f"     - {p['pattern'][:50]}... (reward: {p['reward']})")
        
        # 8. Train (placeholder)
        print("\n8. Triggering training...")
        resp = await client.post(f"{RL_URL}/train")
        train_result = resp.json()
        print(f"   Status: {train_result.get('status')}")
        print(f"   Samples used: {train_result.get('samples_used', 0)}")
        print(f"   Average reward: {train_result.get('average_reward', 0):.2f}")
        print(f"   Hints used: {train_result.get('hints_count', 0)}")
        
        # 9. Test query with learned context
        print("\n9. Testing query with learned context...")
        resp = await client.post(
            f"{MEMORY_URL}/query",
            json={"question": "What did the user prefer?"}
        )
        answer = resp.json()
        print(f"   Answer: {answer.get('answer', '')[:100]}...")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Ingested {len(conversations)} conversations")
        print(f"  - Generated {result.get('training_samples', 0)} training samples")
        print(f"  - Patterns detected: {result.get('patterns', []).__len__()}")
        print(f"  - Training runs: {train_result.get('training_runs_total', 0)}")

if __name__ == "__main__":
    print("\nMake sure both servers are running:")
    print("  Memory: http://localhost:8888")
    print("  RL: http://localhost:30000")
    print("\nStart with: ./start.sh")
    print("\n")
    time.sleep(1)
    
    asyncio.run(test_full_loop())
