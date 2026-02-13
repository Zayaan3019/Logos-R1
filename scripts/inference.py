"""
Inference script for Reason-1 model.

Usage:
    python scripts/inference.py \
        --model_path ./outputs/rl/best_model \
        --question "Sarah has 5 apples and buys 3 more. How many does she have?"
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_default_config, ModelConfig
from src.data.tokenizer import create_tokenizer
from src.models.policy import create_policy
from src.models.reward import create_reward_function
from src.inference.search import create_sampler


def load_model(model_path: str, device: str = "cuda"):
    """
    Load trained Reason-1 model.
    
    Args:
        model_path: Path to saved model
        device: Device to load model to
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    config = get_default_config()
    
    # Create tokenizer
    tokenizer = create_tokenizer(
        model_name=config.model.model_name,
        think_start_token=config.model.think_start_token,
        think_end_token=config.model.think_end_token,
    )
    
    # Create and load model
    model = create_policy(
        config=config.model,
        tokenizer_len=len(tokenizer),
    )
    model.load_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    return model, tokenizer


def solve_math_problem(
    question: str,
    model,
    tokenizer,
    device: str = "cuda",
    use_search: bool = True,
    num_samples: int = 8,
):
    """
    Solve a math problem using Reason-1.
    
    Args:
        question: Math problem to solve
        model: Trained model
        tokenizer: Tokenizer
        device: Device for computation
        use_search: Whether to use Best-of-N search
        num_samples: Number of samples for search
    
    Returns:
        Dictionary with reasoning and answer
    """
    # Format prompt
    prompt = tokenizer.format_prompt(question, include_reasoning_prompt=True)
    
    if use_search:
        # Use Best-of-N sampling
        print(f"\n🔍 Generating {num_samples} candidate solutions...")
        
        reward_fn = create_reward_function()
        sampler = create_sampler(
            model=model,
            tokenizer=tokenizer,
            reward_function=reward_fn,
            strategy="best_of_n",
            num_samples=num_samples,
            device=device,
        )
        
        result = sampler.search(prompt)
        
        print(f"✓ Best solution (score: {result.best_score:.3f}):")
        
        return {
            "question": question,
            "reasoning": result.reasoning,
            "answer": result.answer,
            "score": result.best_score,
            "all_scores": result.all_scores,
        }
    else:
        # Single generation
        print("\n🤔 Generating solution...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )
        
        # Decode output
        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Parse reasoning and answer
        parsed = tokenizer.decode_with_reasoning_sep(
            generated_ids, skip_special_tokens=True
        )
        
        return {
            "question": question,
            "reasoning": parsed.get("reasoning", ""),
            "answer": parsed.get("answer", ""),
            "full_output": generated_text,
        }


def main(args):
    """Main inference function."""
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model_path, device)
    
    # Solve problem
    result = solve_math_problem(
        question=args.question,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_search=args.use_search,
        num_samples=args.num_samples,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("📝 PROBLEM")
    print("=" * 80)
    print(result["question"])
    
    print("\n" + "=" * 80)
    print("🧠 REASONING")
    print("=" * 80)
    print(result["reasoning"])
    
    print("\n" + "=" * 80)
    print("✅ ANSWER")
    print("=" * 80)
    print(result["answer"])
    
    if "score" in result:
        print(f"\n📊 Confidence Score: {result['score']:.3f}")
    
    if "all_scores" in result and len(result["all_scores"]) > 1:
        print(f"📈 All Scores: {[f'{s:.2f}' for s in result['all_scores']]}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with Reason-1")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    
    # Problem arguments
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Math problem to solve"
    )
    
    # Inference arguments
    parser.add_argument(
        "--use_search",
        action="store_true",
        default=True,
        help="Use Best-of-N search (recommended)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples for Best-of-N"
    )
    
    args = parser.parse_args()
    
    main(args)
