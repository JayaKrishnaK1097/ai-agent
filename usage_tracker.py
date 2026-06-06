import os
from dotenv import load_dotenv

load_dotenv()


class UsageTracker:
    """Tracks token usage and costs for the Gemini model."""

    def __init__(self):
        # Gemini pricing — loaded from environment so it can be updated without code changes
        self.input_rate = float(os.getenv("GEMINI_INPUT_COST_PER_MILLION"))
        self.output_rate = float(os.getenv("GEMINI_OUTPUT_COST_PER_MILLION"))
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
    pass

    def extract_token_usage(self, result: dict) -> tuple[int, int]:
        """
        Extract total input/output tokens from an agent invocation result.
        
        The agent's result['messages'] contains multiple message objects.
        AIMessage objects have a usage_metadata field with token counts.
        Sum these across all messages.
        
        Returns (total_input_tokens, total_output_tokens).
        """
        total_input = 0
        total_output = 0
        for message in result.get("messages", []):
            if hasattr(message, "usage_metadata") and message.usage_metadata:
                usage = message.usage_metadata
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
        return total_input, total_output
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost based on token counts and Gemini pricing."""
        costs = (input_tokens * self.input_rate + output_tokens * self.output_rate) / 1_000_000
        return costs

    def record_usage(self, input_tokens: int, output_tokens: int) -> float:
        """
        Record a single LLM call's token usage. Updates lifetime stats.
        Returns the cost of this specific call (so callers can log it per-request).
        """
        cost = self.calculate_cost(input_tokens, output_tokens)
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        return cost

    def get_stats(self) -> dict:
        """Return current lifetime usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd
        }

# Quick standalone test — runs when you execute the file directly
if __name__ == "__main__":
    tracker = UsageTracker()
    
    print("Test 1: empty stats")
    print(f"  {tracker.get_stats()}")
    
    print("\nTest 2: record a fake call (500 input, 200 output)")
    cost = tracker.record_usage(500, 200)
    print(f"  cost of this call: ${cost:.6f}")
    print(f"  stats: {tracker.get_stats()}")
    
    print("\nTest 3: record another call (1000 input, 300 output)")
    cost = tracker.record_usage(1000, 300)
    print(f"  cost of this call: ${cost:.6f}")
    print(f"  stats: {tracker.get_stats()}")