# Objective audit

Production Dynamic-Norm score:
`-850 * norm(cost) + 600 * norm(cumulative_squared_distance) + log_prior`.

Normalization uses epsilon=1e-08 in the denominator and returns zeros when range < 1e-12.
Pool-Norm Exhaustive fixes J_norm and R_D once per filtered pool; required conversion and enumeration are included in its multimodal selection timing.

## Effective raw slopes

```json
{
  "alpha_cost": {
    "count": 880.0,
    "mean": 336.84622443656997,
    "median": 349.0511032218767,
    "p95": 473.5545466846679,
    "min": 89.39009519135017,
    "max": 584.6484812405978
  },
  "alpha_diversity": {
    "count": 880.0,
    "mean": 1247.371837969706,
    "median": 1146.9508623318407,
    "p95": 2477.728722154043,
    "min": 130.9293751095342,
    "max": 3389.2575343097988
  },
  "relative_alpha": {
    "count": 880.0,
    "mean": 3.628276720366214,
    "median": 3.283126299771488,
    "p95": 6.298715082974254,
    "min": 0.8153912710392469,
    "max": 9.060727479247856
  }
}
```

The relative slope is `600 * (cost_range + epsilon) / (850 * (diversity_range + epsilon))` for nondegenerate ranges.