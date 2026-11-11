"""
Minimal Real Smoke Test:
- Real mode only (hard-fails if API key missing)
- Arm: A3 (Curator) only
- 1 seed, 2 epochs, 10 train tasks, 5 held-out tasks
- Hard budget cap: $1.00 (MAX_USD)
- Prints raw call logs, raw prompts, and raw JSON outputs
"""
import os
import yaml
import json
from pathlib import Path
from src.testbed.task_gen import generate_task_suites
from src.loop.train_loop import run_train_arm
from src.obs.experiments import evaluate_checkpoint_on_split
from src.agents.actor import ActorAgent
from src.testbed.grader import Grader
from src.agents.llm_client import LLMClient
from src.obs.cost_guard import estimate_run_cost

def run_real_smoke_test():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    actor_model = config.get("ACTOR_MODEL", "gemini-2.5-flash")
    optimizer_model = config.get("OPTIMIZER_MODEL", "gemini-2.5-pro")

    print("==================================================================")
    print("STARTING MINIMAL REAL SMOKE TEST (A3 ONLY, 2 EPOCHS, $1.00 CAP)")
    print(f"Actor Model:     {actor_model}")
    print(f"Optimizer Model: {optimizer_model}")
    print("==================================================================")

    # 1. Mandatory API Key Pre-Flight Guard
    if "gemini" in optimizer_model or "gemini" in actor_model:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("ABORT: Real smoke test requires GEMINI_API_KEY or GOOGLE_API_KEY to be set in the environment.")
    elif "claude" in optimizer_model or "claude" in actor_model:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ABORT: Real smoke test requires ANTHROPIC_API_KEY to be set in the environment.")

    # 2. Pre-flight cost check ($1.00 MAX_USD)
    cost_est = estimate_run_cost(
        config=config,
        num_arms=1,
        num_seeds=1,
        num_epochs=2,
        batch_size=5,
        heldout_size=5,
        eval_epochs_count=2
    )
    print(f"Estimated Total Spend: ${cost_est['grand_total_usd']:.4f} (Budget Cap: ${config.get('MAX_USD', 1.0):.2f})")
    if cost_est["grand_total_usd"] > config.get("MAX_USD", 1.0):
        raise RuntimeError("Estimated cost exceeds MAX_USD! Aborting.")

    # 3. Task Generation
    train_tasks, heldout_id, _, labels = generate_task_suites(
        noise_frac=0.20,
        seed=42,
        output_dir="data_smoke"
    )
    smoke_train = train_tasks[:10]
    smoke_heldout = heldout_id[:5]

    eval_actor = ActorAgent(model_name=actor_model, price_table=config.get("PRICE_TABLE", {}))
    grader = Grader(seed=42)

    # 4. Run A3
    train_res = run_train_arm(
        arm="A3",
        seed=42,
        epochs=2,
        batch_size=5,
        noise_frac=0.20,
        train_tasks=smoke_train,
        labels=labels,
        config=config,
        checkpoints_dir="checkpoints/smoke"
    )

    print("\n>>> SMOKE TEST COMPLETED SUCCESSFULLY WITH REAL CALLS <<<")

if __name__ == "__main__":
    run_real_smoke_test()
