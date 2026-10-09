# Objective audit

Production Dynamic-Norm score:
`-850 * norm(cost) + 600 * norm(cumulative_squared_distance) + log_prior`.

Normalization uses epsilon=1e-08 in the denominator and returns zeros when range < 1e-12.
Pool-Norm Exhaustive fixes J_norm and R_D once per filtered pool; required conversion and enumeration are included in its multimodal selection timing.

## Effective raw slopes

```json
{
  "alpha_cost": {
    "count": 8750.0,
    "mean": 334.323398628715,
    "median": 348.14891492688577,
    "p95": 480.75925476404086,
    "min": 61.08575451378615,
    "max": 700.2958143278018
  },
  "alpha_diversity": {
    "count": 8750.0,
    "mean": 1213.7444238100034,
    "median": 1091.7101673560537,
    "p95": 2490.746714935908,
    "min": 29.015087940777683,
    "max": 4267.57816983117
  },
  "relative_alpha": {
    "count": 8750.0,
    "mean": 3.5137416947669204,
    "median": 3.157950624810703,
    "p95": 6.271150442531452,
    "min": 0.36493525947111105,
    "max": 9.723245329129035
  }
}
```

The relative slope is `600 * (cost_range + epsilon) / (850 * (diversity_range + epsilon))` for nondegenerate ranges.