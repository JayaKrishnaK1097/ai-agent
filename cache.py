"""
Response cache for AI Agent.
Stores the question -> answer pairs with a configurable TTL (time-to-live).
Used to avoid redundent LLM calls for repeated questions, improving performance and reducing costs.
Theoretically, 40% of the questions processed by Claude, Gemini, and other LLMs are repeated and can be answered from cache instead of making a new API call.
"""

import time
from typing import Any, Optional

class ResponseCache:
    # Initialize the cache with an optional TTL (time-to-live) in seconds.
    def __init__(self, ttl: Optional[int] = 300):
        self.cache = {}
        self.ttl = ttl if ttl else 300
        self.hit_count = 0
        self.miss_count = 0
        pass
    
    def get(self, key: str) -> Optional[Any]:
        # Check if the key exists in the cache and is not expired.
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry["expires_at"]:
                self.hit_count += 1
                return entry["value"]
            else:
                # Cache entry expired, remove it
                del self.cache[key]
        self.miss_count += 1
        return None

    def set(self, key: str, value: Any) -> None:
        # Store the value in the cache with the current timestamp.
        self.cache[key] = {"value": value, "expires_at": time.time() + self.ttl}

    def stats(self) -> dict:
        # Return cache statistics including hit count, miss count, and hit rate.
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total) * 100 if total > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "current_cache_size": len(self.cache)
        }

# Test block - runs when executing this file directly "Python cache.py"
if __name__ == "__main__":
    cache = ResponseCache(ttl=10)  # Cache entries expire after 10 seconds

    # Test setting and getting values
    print("Test 1: empty set returns None for missing key")
    question = "Hello"
    result = cache.get(question)  # Should return None (miss)
    print(f" question: {question}")
    print(f" result: {result}\n")

    print("Test 2: set and get a return value")
    cache.set(question, "World")
    result = cache.get(question)  # Should return "World" (hit)
    print(f" question: {question}")
    print(f" result: {result}\n")

    print("Test 3: stats after 1 hit and 1 miss")
    print(cache.stats())  # Should show hit_count=1, miss_count=1, hit_rate=50.0%
    
    print("Test 4: get key after some time, but before TTL expires")
    time.sleep(5)
    result = cache.get(question)  # Should still return "World" (hit, not expired)
    print(f" question: {question}")
    print(f" result: {result}\n")

    print("Test 5: get key after TTL expires")
    time.sleep(6)
    result = cache.get(question)  # Should return None (miss, expired)
    print(f" question: {question}")
    print(f" result: {result}\n")

    print("Test 6: get a different key that was never set")
    question = "Bye"
    result = cache.get(question)  # Should return None (miss, never set) 
    print(f" question: {question}")
    print(f" result: {result}\n")

    print("Test 7: cache statistics")
    # Test statistics
    print(cache.stats())  # Should show hit_count=2, miss_count=2, hit_rate=40.0%