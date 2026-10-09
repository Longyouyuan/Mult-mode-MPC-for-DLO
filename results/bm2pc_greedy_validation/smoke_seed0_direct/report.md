# BM2PC Greedy Validation Report

Smoke run: `False`

## Objective fidelity

Dynamic-Norm Greedy uses the production cost min-max normalization and per-step dynamic diversity normalization. Fixed-F Greedy and Fixed-F Exhaustive use the fixed reference objective with raw rollout costs, log prior, raw squared Euclidean features, and beta=600.

## Offline comparison

`gap_fixed_greedy` is the fixed-F greedy approximation gap. `normalization_effect = F_fixed_greedy - F_dynamic` combines cost normalization and dynamic diversity normalization effects.

## Closed-loop comparison

Closed-loop values compare three selection rules under matched initial seeds. Later candidate pools can diverge because selected warm starts and states diverge.

## Certificate

The certificate is a finite filtered-pool bound for each returned set. It is not an approximation-ratio or stability guarantee.

## Aggregate results

```json
{
  "controllers": {
    "dynamic": {
      "episodes": 1.0,
      "rmse_mean": 0.007427572012680244,
      "rmse_std": 0.0,
      "collision_free_rate": 1.0,
      "planning_mean_ms": 18.316559317000525,
      "planning_p95_mean_ms": 24.705520013230853,
      "deadline_miss_rate_mean": 0.045454545454545456
    },
    "fixed-exhaustive": {
      "episodes": 1.0,
      "rmse_mean": 0.007411827430866892,
      "rmse_std": 0.0,
      "collision_free_rate": 1.0,
      "planning_mean_ms": 100.6266877278077,
      "planning_p95_mean_ms": 125.42012498597613,
      "deadline_miss_rate_mean": 0.9931818181818182
    },
    "fixed-greedy": {
      "episodes": 1.0,
      "rmse_mean": 0.00732952637515741,
      "rmse_std": 0.0,
      "collision_free_rate": 1.0,
      "planning_mean_ms": 15.871154319161592,
      "planning_p95_mean_ms": 21.00984502321807,
      "deadline_miss_rate_mean": 0.01818181818181818
    }
  },
  "offline": {
    "snapshots": 440.0,
    "dynamic_gap_mean": 144.67434631560974,
    "dynamic_gap_p95": 822.8686077038824,
    "dynamic_gap_max": 2172.277186889887,
    "fixed_greedy_gap_mean": 0.35358826953947753,
    "fixed_greedy_gap_p95": 0.0,
    "fixed_greedy_gap_max": 66.90815039160407,
    "normalization_effect_mean": 144.32075804607024,
    "normalization_effect_min": -66.90815039160407,
    "normalization_effect_max": 2172.277186889887,
    "dynamic_bound_violations": 0.0,
    "fixed_greedy_bound_violations": 0.0,
    "fixed_greedy_exact_frequency": 0.9840909090909091
  }
}
```

## Closed-loop metrics

```json
[
  {
    "controller": "dynamic",
    "seed": 0,
    "steps": 440,
    "tracking_rmse": 0.007427572012680244,
    "tracking_mean": 0.006958193775332101,
    "tracking_max": 0.014376944715407661,
    "collision_count": 0,
    "wall_mean_ms": 18.316559317000525,
    "wall_median_ms": 16.321000002790242,
    "wall_p95_ms": 24.705520013230853,
    "deadline_miss_rate": 0.045454545454545456,
    "fallback_count": 0
  },
  {
    "controller": "fixed-exhaustive",
    "seed": 0,
    "steps": 440,
    "tracking_rmse": 0.007411827430866892,
    "tracking_mean": 0.0069553662387141155,
    "tracking_max": 0.012974152422462968,
    "collision_count": 0,
    "wall_mean_ms": 100.6266877278077,
    "wall_median_ms": 99.73624999111053,
    "wall_p95_ms": 125.42012498597613,
    "deadline_miss_rate": 0.9931818181818182,
    "fallback_count": 0
  },
  {
    "controller": "fixed-greedy",
    "seed": 0,
    "steps": 440,
    "tracking_rmse": 0.00732952637515741,
    "tracking_mean": 0.006894460170793542,
    "tracking_max": 0.012330300412610351,
    "collision_count": 0,
    "wall_mean_ms": 15.871154319161592,
    "wall_median_ms": 14.195700001437217,
    "wall_p95_ms": 21.00984502321807,
    "deadline_miss_rate": 0.01818181818181818,
    "fallback_count": 0
  }
]
```

## Offline summary

```json
[
  {
    "seed": "0",
    "step": "0",
    "M": "200",
    "F_dynamic": "-15457.483372122631",
    "F_fixed_greedy": "-15457.483372122631",
    "F_exact": "-15457.483372122631",
    "F_min": "-22823.454056776915",
    "gap_dynamic": "0.0",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "0.0",
    "bound_dynamic": "67.83965371680824",
    "bound_fixed_greedy": "67.83965371680824",
    "slack_dynamic": "67.83965371680824",
    "slack_fixed_greedy": "67.83965371680824",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "1",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "1",
    "M": "200",
    "F_dynamic": "-14113.135729100377",
    "F_fixed_greedy": "-13578.582670184343",
    "F_exact": "-13578.582670184343",
    "F_min": "-19684.343446153416",
    "gap_dynamic": "534.5530589160335",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "534.5530589160335",
    "bound_dynamic": "589.1302499088088",
    "bound_fixed_greedy": "0.0",
    "slack_dynamic": "54.577190992775286",
    "slack_fixed_greedy": "0.0",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "0",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "2",
    "M": "200",
    "F_dynamic": "-13430.898524187349",
    "F_fixed_greedy": "-13400.189518154086",
    "F_exact": "-13400.189518154086",
    "F_min": "-18386.74631214946",
    "gap_dynamic": "30.709006033262995",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "30.709006033262995",
    "bound_dynamic": "37.94107223062019",
    "bound_fixed_greedy": "0.0",
    "slack_dynamic": "7.232066197357199",
    "slack_fixed_greedy": "0.0",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "0",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "3",
    "M": "200",
    "F_dynamic": "-12903.031948894595",
    "F_fixed_greedy": "-12903.031948894595",
    "F_exact": "-12903.031948894595",
    "F_min": "-17404.08169480228",
    "gap_dynamic": "0.0",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "0.0",
    "bound_dynamic": "0.0",
    "bound_fixed_greedy": "0.0",
    "slack_dynamic": "0.0",
    "slack_fixed_greedy": "0.0",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "1",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "4",
    "M": "200",
    "F_dynamic": "-12876.513984577987",
    "F_fixed_greedy": "-12876.513984577987",
    "F_exact": "-12876.513984577987",
    "F_min": "-17952.462396905925",
    "gap_dynamic": "0.0",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "0.0",
    "bound_dynamic": "0.0",
    "bound_fixed_greedy": "0.0",
    "slack_dynamic": "0.0",
    "slack_fixed_greedy": "0.0",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "1",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "5",
    "M": "200",
    "F_dynamic": "-13992.411589168487",
    "F_fixed_greedy": "-13992.411589168487",
    "F_exact": "-13992.411589168487",
    "F_min": "-19341.368805066042",
    "gap_dynamic": "0.0",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "0.0",
    "bound_dynamic": "0.1208938123054395",
    "bound_fixed_greedy": "0.1208938123054395",
    "slack_dynamic": "0.1208938123054395",
    "slack_fixed_greedy": "0.1208938123054395",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "1",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "6",
    "M": "200",
    "F_dynamic": "-13569.230149340214",
    "F_fixed_greedy": "-13569.230149340214",
    "F_exact": "-13569.230149340214",
    "F_min": "-19111.09060848168",
    "gap_dynamic": "0.0",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "0.0",
    "bound_dynamic": "0.0",
    "bound_fixed_greedy": "0.0",
    "slack_dynamic": "0.0",
    "slack_fixed_greedy": "0.0",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "1",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "7",
    "M": "200",
    "F_dynamic": "-12796.085672848736",
    "F_fixed_greedy": "-12796.085672848736",
    "F_exact": "-12796.085672848736",
    "F_min": "-19080.55543812556",
    "gap_dynamic": "0.0",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "0.0",
    "bound_dynamic": "0.0",
    "bound_fixed_greedy": "0.0",
    "slack_dynamic": "0.0",
    "slack_fixed_greedy": "0.0",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "1",
    "exact_fixed_greedy": "1"
  },
  {
    "seed": "0",
    "step": "8",
    "M": "200",
    "F_dynamic": "-12507.842458706928",
    "F_fixed_greedy": "-12507.842458706928",
    "F_exact": "-12507.324057860618",
    "F_min": "-17455.84819113669",
    "gap_dynamic": "0.518400846309305",
    "gap_fixed_greedy": "0.518400846309305",
    "normalization_effect": "0.0",
    "bound_dynamic": "23.499821986733878",
    "bound_fixed_greedy": "23.499821986733878",
    "slack_dynamic": "22.981421140424573",
    "slack_fixed_greedy": "22.981421140424573",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "0",
    "exact_fixed_greedy": "0"
  },
  {
    "seed": "0",
    "step": "9",
    "M": "200",
    "F_dynamic": "-13614.484526481801",
    "F_fixed_greedy": "-13591.13564235445",
    "F_exact": "-13591.13564235445",
    "F_min": "-19290.526339333337",
    "gap_dynamic": "23.348884127351994",
    "gap_fixed_greedy": "0.0",
    "normalization_effect": "23.348884127351994",
    "bound_dynamic": "79.52352646899453",
    "bound_fixed_greedy": "58.48426627801018",
    "slack_dynamic": "56.17464234164254",
    "slack_fixed_greedy": "58.48426627801018",
    "dynamic_bound_ok": "1",
    "fixed_greedy_bound_ok": "1",
    "exact_dynamic": "0",
    "exact_fixed_greedy": "1"
  }
]
```
