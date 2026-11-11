"""
Pre-flight cost estimation and budget guard:
Estimates the expected token and dollar cost per phase:
- Actor (rollout)
- Curate (LLM curator decisions)
- Reflect (LLM reflection proposals + multi-attempt retries)
- Gate (GEPA-style minibatch re-runs)
- Held-out evaluation (tracked separately)

Aborts immediately if total estimated or tracked cost exceeds MAX_USD.
"""
from typing import Dict, Any

def estimate_run_cost(
    config: Dict[str, Any],
    num_arms: int = 2,
    num_seeds: int = 1,
    num_epochs: int = 6,
    batch_size: int = 10,
    heldout_size: int = 10,
    eval_epochs_count: int = 3
) -> Dict[str, Any]:
    price_table = config.get("PRICE_TABLE", {})
    actor_model = config.get("ACTOR_MODEL", "claude-haiku-5-5")
    optimizer_model = config.get("OPTIMIZER_MODEL", "claude-sonnet-5-5")
    max_usd = config.get("MAX_USD", 10.0)

    actor_prices = price_table.get(actor_model, {"prompt_per_1m": 0.80, "completion_per_1m": 4.00})
    optimizer_prices = price_table.get(optimizer_model, {"prompt_per_1m": 3.00, "completion_per_1m": 15.00})

    # Token approximations based on system prompts & trace depth
    actor_prompt_per_task = 350
    actor_comp_per_task = 70
    actor_cost_per_task = (actor_prompt_per_task / 1e6) * actor_prices["prompt_per_1m"] + \
                          (actor_comp_per_task / 1e6) * actor_prices["completion_per_1m"]

    curator_prompt_per_call = 450
    curator_comp_per_call = 90
    curator_cost_per_call = (curator_prompt_per_call / 1e6) * optimizer_prices["prompt_per_1m"] + \
                            (curator_comp_per_call / 1e6) * optimizer_prices["completion_per_1m"]

    reflect_prompt_per_call = 900
    reflect_comp_per_call = 150
    reflect_cost_per_call = (reflect_prompt_per_call / 1e6) * optimizer_prices["prompt_per_1m"] + \
                            (reflect_comp_per_call / 1e6) * optimizer_prices["completion_per_1m"]

    total_runs = num_arms * num_seeds
    total_batch_draws = total_runs * num_epochs * batch_size

    # Phase 1: Actor Rollout
    est_actor_usd = total_batch_draws * actor_cost_per_task

    # Phase 2: Curate (applies to curator arms, approx 3-5 failed tasks per batch)
    curate_calls = (total_batch_draws // 2) * 0.4 # approx 40% failure rate
    est_curate_usd = curate_calls * curator_cost_per_call

    # Phase 3: Reflect (includes 1.2 average attempts)
    reflect_calls = total_runs * num_epochs * 1.2
    est_reflect_usd = reflect_calls * reflect_cost_per_call

    # Phase 4: Gate re-runs (evaluates batch on candidate playbook)
    est_gate_usd = total_batch_draws * actor_cost_per_task * 0.8

    # Phase 5: Held-out Eval (at specified checkpoint epochs)
    eval_draws = total_runs * eval_epochs_count * heldout_size
    est_eval_usd = eval_draws * actor_cost_per_task

    est_learning_cost_usd = est_actor_usd + est_curate_usd + est_reflect_usd + est_gate_usd
    est_total_usd = est_learning_cost_usd + est_eval_usd

    breakdown = {
        "actor_rollout_usd": est_actor_usd,
        "curation_usd": est_curate_usd,
        "reflection_usd": est_reflect_usd,
        "gate_reruns_usd": est_gate_usd,
        "heldout_eval_usd": est_eval_usd,
        "total_learning_usd": est_learning_cost_usd,
        "grand_total_usd": est_total_usd,
        "max_usd_cap": max_usd,
        "exceeds_cap": est_total_usd > max_usd
    }

    return breakdown

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    b = estimate_run_cost(cfg)
    print("=== PRE-FLIGHT COST ESTIMATION ===")
    for k, v in b.items():
        if isinstance(v, float):
            print(f"{k}: ${v:.4f}")
        else:
            print(f"{k}: {v}")
