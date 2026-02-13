"""
Example usage script demonstrating Reason-1 capabilities.
"""

# Example 1: Simple Math Problem
print("="*80)
print("Example 1: Basic Addition")
print("="*80)
print("""
python scripts/inference.py \\
    --model_path ./outputs/rl/best_model \\
    --question "John has 5 apples and buys 3 more. How many apples does he have now?" \\
    --use_search \\
    --num_samples 8

Expected Output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: John starts with 5 apples.
Step 2: He buys 3 more apples.
Step 3: To find the total, I need to add: 5 + 3 = 8
Step 4: Therefore, John has 8 apples now.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ANSWER: 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Example 2: Multi-Step Problem
print("\n" + "="*80)
print("Example 2: Multi-Step Problem")
print("="*80)
print("""
python scripts/inference.py \\
    --model_path ./outputs/rl/best_model \\
    --question "A store has 120 items. They sell 45 in the morning and 32 in the afternoon. How many items are left?" \\
    --use_search \\
    --num_samples 8

Expected Reasoning:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Start with total items = 120
Step 2: Items sold in morning = 45
Step 3: Items sold in afternoon = 32
Step 4: Total items sold = 45 + 32 = 77
Step 5: Items remaining = 120 - 77 = 43
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ANSWER: 43
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Example 3: Self-Correction
print("\n" + "="*80)
print("Example 3: Self-Correction (Advanced)")
print("="*80)
print("""
python scripts/inference.py \\
    --model_path ./outputs/rl/best_model \\
    --question "If 4 shirts cost $60, how much do 7 shirts cost?" \\
    --use_search \\
    --num_samples 8

Expected Reasoning (with self-correction):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Find price per shirt: $60 ÷ 4 = $15
Step 2: Calculate cost for 7 shirts: $15 × 7 = $105
Wait, let me double-check this calculation:
- Price per shirt: 60/4 = 15 ✓
- Cost for 7: 15 × 7 = 105 ✓
The answer is correct.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ANSWER: $105
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Python API Usage
print("\n" + "="*80)
print("Python API Usage")
print("="*80)
print("""
from src.models.policy import create_policy
from src.data.tokenizer import create_tokenizer
from src.inference.search import create_sampler
from src.models.reward import create_reward_function

# Load model
model = create_policy(config, tokenizer_len)
model.load_pretrained("./outputs/rl/best_model")

# Create tokenizer
tokenizer = create_tokenizer(model_name="meta-llama/Llama-3-8B")

# Create sampler
reward_fn = create_reward_function()
sampler = create_sampler(
    model=model,
    tokenizer=tokenizer,
    reward_function=reward_fn,
    strategy="best_of_n",
    num_samples=8,
)

# Solve problem
result = sampler.search(
    prompt="What is 15% of 200?",
)

print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning}")
print(f"Confidence: {result.best_score}")
""")

print("\n" + "="*80)
print("For more examples, see README.md")
print("="*80)
