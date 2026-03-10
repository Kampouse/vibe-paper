"""Integrated Agent - Combines Memory + RL
Lightweight version without FastAPI overhead for testing
"""
import os
import asyncio
import httpx

MEMORY_URL = os.getenv("MEMORY_URL", "http://localhost:8888")
RL_URL = os.getenv("RL_URL", "http://localhost:30000")

class IntegratedAgent:
    """Agent with memory and learning"""
    
    def __init__(self, memory_url: str = MEMORY_URL, rl_url: str = RL_URL):
        self.memory_url = memory_url
        self.rl_url = rl_url
        self.session_id = "default"
    
    async def chat(self, message: str, use_memory: bool = True) -> dict:
        """Send a message and get response with memory context"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get context from memory
            context = ""
            if use_memory:
                try:
                    resp = await client.post(
                        f"{self.memory_url}/query",
                        json={"question": message, "include_insights": True}
                    )
                    context = resp.json().get("answer", "")
                except:
                    pass
            
            # Build messages
            messages = [{"role": "user", "content": message}]
            if context:
                messages.insert(0, {
                    "role": "system",
                    "content": f"[Context from memory]\n{context}"
                })
            
            # Generate response
            resp = await client.post(
                f"{self.rl_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "session_id": self.session_id
                }
            )
            data = resp.json()
            response = data["choices"][0]["message"]["content"]
            
            # Store in memory
            if use_memory:
                try:
                    await client.post(
                        f"{self.memory_url}/ingest",
                        json={
                            "text": f"User: {message}\nAssistant: {response}",
                            "source": f"session/{self.session_id}"
                        }
                    )
                except:
                    pass
            
            return {
                "response": response,
                "context_used": bool(context),
                "session_id": self.session_id
            }
    
    async def give_feedback(self, reward: float, hint: str = None):
        """Give feedback on last response"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{self.rl_url}/feedback",
                json={
                    "session_id": self.session_id,
                    "turn_id": 0,  # Last turn
                    "reward": reward,
                    "hint": hint
                }
            )
    
    async def consolidate(self) -> dict:
        """Trigger memory consolidation (generates training samples)"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.memory_url}/consolidate")
            return resp.json()
    
    async def get_stats(self) -> dict:
        """Get stats from both services"""
        stats = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{self.memory_url}/stats")
                stats["memory"] = resp.json()
            except:
                stats["memory"] = {"error": "unreachable"}
            
            try:
                resp = await client.get(f"{self.rl_url}/samples")
                stats["rl"] = resp.json()
            except:
                stats["rl"] = {"error": "unreachable"}
        
        return stats

# CLI interface
async def main():
    import sys
    
    agent = IntegratedAgent()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agent.py chat 'your message'")
        print("  python agent.py feedback 1.0 'good response'")
        print("  python agent.py consolidate")
        print("  python agent.py stats")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "chat":
        message = sys.argv[2] if len(sys.argv) > 2 else "Hello"
        result = await agent.chat(message)
        print(f"\nResponse: {result['response']}")
        print(f"Context used: {result['context_used']}")
    
    elif cmd == "feedback":
        reward = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        hint = sys.argv[3] if len(sys.argv) > 3 else None
        await agent.give_feedback(reward, hint)
        print(f"Feedback sent: reward={reward}, hint={hint}")
    
    elif cmd == "consolidate":
        result = await agent.consolidate()
        print(f"Consolidation result:")
        print(f"  Memories processed: {result.get('memories_processed', 0)}")
        print(f"  Insights created: {result.get('insights_created', 0)}")
        print(f"  Training samples: {result.get('training_samples', 0)}")
        if result.get('patterns'):
            print(f"\nPatterns detected:")
            for p in result['patterns']:
                print(f"  - [{p['type']}] {p['pattern']} (reward: {p['reward']})")
    
    elif cmd == "stats":
        stats = await agent.get_stats()
        print(f"Memory: {stats['memory']}")
        print(f"RL: {stats['rl']}")
    
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    asyncio.run(main())
