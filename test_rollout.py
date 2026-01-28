from calc_next_token_distrib import monte_carlo_rollouts

per_step_probs, sampled = monte_carlo_rollouts(
    "Generate 4 random integers between 1 and 100:",
    steps=4,
    n_samples=1,
    tracked_integers=range(1, 101),
)
print("sampled token ids:", sampled[0])
print("per-step prob for 7 at step 0:", per_step_probs[0][7])
