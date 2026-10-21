| cell | policy_mode | scenario_id | n_seeds | backend | gossip_count_mean | gossip_rate_per_event_mean | belief_accuracy_mean | belief_brier_mean | llm_invalid_action_rate_mean | llm_act_attempts_mean | llm_act_successes_mean | llm_act_fallbacks_mean | llm_act_repairs_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | llm_openai | toy_provenance_stress_v0 | 5 | openai | 0.000 | 0.000 | 0.629 | 0.250 | 0.000 | 70.000 | 70.000 | 0.000 | 0.000 |
| a | reference | toy_provenance_stress_v0 | 5 | openai | 24.000 | 0.343 | 0.629 | 0.250 |  |  |  |  |  |
| a | utility_only | toy_provenance_stress_v0 | 5 | openai | 23.800 | 0.340 | 0.629 | 0.250 |  |  |  |  |  |
| b | llm_openai | toy_provenance_stress_matched_gossip_v0 | 5 | openai | 0.000 | 0.000 | 0.626 | 0.243 | 0.000 | 70.000 | 70.000 | 0.000 | 0.000 |
| b | reference | toy_provenance_stress_matched_gossip_v0 | 5 | openai | 22.200 | 0.317 | 0.543 | 0.261 |  |  |  |  |  |
| b | utility_only | toy_provenance_stress_matched_gossip_v0 | 5 | openai | 23.600 | 0.337 | 0.543 | 0.263 |  |  |  |  |  |
