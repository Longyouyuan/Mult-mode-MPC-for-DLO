# BM2PC Greedy Validation Report

Smoke run: `True`

## Objective fidelity

Dynamic-Norm Greedy calls production Planner._greedy_select() unchanged. Pool-Norm Exhaustive fixes one cost normalization and one diversity scale per filtered pool, then globally maximizes F_norm over all triples. It is not the exact optimum of the production stepwise score.

## Offline comparison

The reported gap is F_norm(S_exact)-F_norm(G_dynamic); Delta_G is the a-posteriori certificate using exactly the same pool-normalized weights and beta_eff. Normalized bounds are not clipped and can exceed one; zero objective ranges are excluded from normalized statistics.

## Closed-loop comparison

`tracking_rmse` is the free rope tip (`S_first`) error relative to the time-aligned Lemniscate reference `goal_builder.get(step + 1)`, measured after executing that control interval. `actuator_endpoint_rmse` separately reports the attached endpoint (`S_last`) error relative to the filtered actuator command.

Closed-loop values compare two selection rules under matched initial seeds. Tracking is free-tip S_first versus goal_builder.get(step+1) after the control interval. Collision events and substeps scan every MuJoCo substep.

## Certificate

The certificate is a finite filtered-pool bound for each returned set. It is not an approximation-ratio or stability guarantee.

## Aggregate results

```json
{
  "controllers": {
    "dynamic": {
      "episodes": 1.0,
      "rmse_mean": 0.057862682244815986,
      "rmse_std": 0.0,
      "actuator_endpoint_rmse_mean": 0.007427572012680244,
      "collision_free_rate": 0.0,
      "collision_event_mean": 0.0,
      "collision_substeps_mean": 0.0,
      "planning_mean_ms": 15.533291364937957,
      "planning_p95_mean_ms": 19.53308000956895,
      "deadline_miss_rate_mean": 0.0022727272727272726,
      "fallback_steps_total": 0.0
    },
    "pool-norm-exhaustive": {
      "episodes": 1.0,
      "rmse_mean": 0.05441153916532642,
      "rmse_std": 0.0,
      "actuator_endpoint_rmse_mean": 0.007286766732735229,
      "collision_free_rate": 0.0,
      "collision_event_mean": 0.0,
      "collision_substeps_mean": 0.0,
      "planning_mean_ms": 102.92566613663273,
      "planning_p95_mean_ms": 122.41008500132011,
      "deadline_miss_rate_mean": 0.9977272727272727,
      "fallback_steps_total": 0.0
    }
  },
  "offline": {
    "snapshots": 440.0,
    "gap_mean": 24.09934721912825,
    "gap_median": 0.0,
    "gap_p95": 161.2131349720161,
    "normalized_gap_mean": 0.010004479003849773,
    "normalized_gap_median": 0.0,
    "normalized_gap_p95": 0.06734611349896413,
    "bound_mean": 113.52266929567847,
    "bound_median": 26.608971578692945,
    "bound_p95": 518.8440289695735,
    "normalized_bound_mean": 0.04736372996660899,
    "normalized_bound_median": 0.010885499937771118,
    "normalized_bound_p95": 0.2240913624312329,
    "exact_optimal_frequency": 0.5931818181818181,
    "bound_violations": 0.0,
    "objective_range_degenerate_count": 0.0
  }
}
```

## Closed-loop metrics

```json
[
  {
    "controller": "dynamic",
    "seed": "0",
    "steps": "440",
    "tracking_rmse": "0.057862682244815986",
    "tracking_mean": "0.04485479908782238",
    "tracking_max": "0.16542437376297556",
    "actuator_endpoint_rmse": "0.007427572012680244",
    "actuator_endpoint_mean": "0.006958193775332101",
    "actuator_endpoint_max": "0.014376944715407661",
    "collision_count": "0",
    "collision_substeps": "0",
    "collision_any": "False",
    "wall_mean_ms": "15.533291364937957",
    "wall_median_ms": "13.461349997669458",
    "wall_p95_ms": "19.53308000956895",
    "deadline_miss_rate": "0.0022727272727272726",
    "fallback_count": "0"
  },
  {
    "controller": "pool-norm-exhaustive",
    "seed": "0",
    "steps": "440",
    "tracking_rmse": "0.05441153916532642",
    "tracking_mean": "0.04036616240317171",
    "tracking_max": "0.15743373414796052",
    "actuator_endpoint_rmse": "0.007286766732735229",
    "actuator_endpoint_mean": "0.006880003947930138",
    "actuator_endpoint_max": "0.01274553477623661",
    "collision_count": "0",
    "collision_substeps": "0",
    "collision_any": "False",
    "wall_mean_ms": "102.92566613663273",
    "wall_median_ms": "101.74715000903234",
    "wall_p95_ms": "122.41008500132011",
    "deadline_miss_rate": "0.9977272727272727",
    "fallback_count": "0"
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
    "F_dynamic": "-661.5777847614861",
    "F_exact": "-661.5777847614861",
    "F_min": "-3038.063857162786",
    "objective_range": "2376.4860724012997",
    "gap": "0.0",
    "raw_gap": "0.0",
    "Delta_G": "65.52518533833245",
    "slack": "65.52518533833245",
    "normalized_gap": "0.0",
    "normalized_bound": "0.027572299328530503",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "3.133371353149414",
    "R_D": "1.4671374433841278",
    "beta_eff": "408.95963675116354",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "1",
    "M": "200",
    "F_dynamic": "-691.3615545848929",
    "F_exact": "-578.2175957432935",
    "F_min": "-3034.5676267720687",
    "objective_range": "2456.3500310287754",
    "gap": "113.14395884159944",
    "raw_gap": "113.14395884159944",
    "Delta_G": "163.90957950403026",
    "slack": "50.76562066243082",
    "normalized_gap": "0.04606182238376351",
    "normalized_bound": "0.06672891787958297",
    "exact_optimal": "0",
    "bound_ok": "1",
    "cost_range": "2.4972734451293945",
    "R_D": "1.0750817142182485",
    "beta_eff": "558.097106930446",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "2",
    "M": "200",
    "F_dynamic": "-531.5218824483037",
    "F_exact": "-531.5218824483037",
    "F_min": "-2910.684324782709",
    "objective_range": "2379.1624423344056",
    "gap": "0.0",
    "raw_gap": "0.0",
    "Delta_G": "6.810689921787343",
    "slack": "6.810689921787343",
    "normalized_gap": "0.0",
    "normalized_bound": "0.0028626418274763856",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "2.0803146362304688",
    "R_D": "1.4532540090807382",
    "beta_eff": "412.86656848851",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "3",
    "M": "200",
    "F_dynamic": "-571.1988020993207",
    "F_exact": "-571.1988020993208",
    "F_min": "-2996.4427683178596",
    "objective_range": "2425.243966218539",
    "gap": "0.0",
    "raw_gap": "-1.1368683772161603e-13",
    "Delta_G": "0.0",
    "slack": "0.0",
    "normalized_gap": "0.0",
    "normalized_bound": "0.0",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "1.8416075706481934",
    "R_D": "0.862082983257431",
    "beta_eff": "695.9886747479669",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "4",
    "M": "200",
    "F_dynamic": "-625.7091469245858",
    "F_exact": "-625.7091469245858",
    "F_min": "-2956.3839005439927",
    "objective_range": "2330.674753619407",
    "gap": "0.0",
    "raw_gap": "0.0",
    "Delta_G": "9.402941786504016",
    "slack": "9.402941786504016",
    "normalized_gap": "0.0",
    "normalized_bound": "0.0040344289875289445",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "2.1520004272460938",
    "R_D": "0.7791417283804769",
    "beta_eff": "770.0781134471878",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "5",
    "M": "200",
    "F_dynamic": "-486.3632328956853",
    "F_exact": "-486.3632328956853",
    "F_min": "-3024.7424254980842",
    "objective_range": "2538.3791926023987",
    "gap": "0.0",
    "raw_gap": "0.0",
    "Delta_G": "2.104361204214001",
    "slack": "2.104361204214001",
    "normalized_gap": "0.0",
    "normalized_bound": "0.0008290176701521756",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "2.129791259765625",
    "R_D": "1.217098876158465",
    "beta_eff": "492.9755559088407",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "6",
    "M": "200",
    "F_dynamic": "-580.3069155836258",
    "F_exact": "-580.3069155836258",
    "F_min": "-3014.924406109753",
    "objective_range": "2434.617490526127",
    "gap": "0.0",
    "raw_gap": "0.0",
    "Delta_G": "4.2971611766970454",
    "slack": "4.2971611766970454",
    "normalized_gap": "0.0",
    "normalized_bound": "0.0017650251809241781",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "2.2588891983032227",
    "R_D": "1.2485963908525228",
    "beta_eff": "480.53958796479714",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "7",
    "M": "200",
    "F_dynamic": "-558.1480722277324",
    "F_exact": "-558.1480722277325",
    "F_min": "-3044.2016150689265",
    "objective_range": "2486.053542841194",
    "gap": "0.0",
    "raw_gap": "-1.1368683772161603e-13",
    "Delta_G": "0.0",
    "slack": "0.0",
    "normalized_gap": "0.0",
    "normalized_bound": "0.0",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "2.541116237640381",
    "R_D": "1.2363323675526283",
    "beta_eff": "485.30638758140844",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "8",
    "M": "200",
    "F_dynamic": "-576.3206547495602",
    "F_exact": "-573.141050465681",
    "F_min": "-2931.92387504363",
    "objective_range": "2358.7828245779488",
    "gap": "3.1796042838791436",
    "raw_gap": "3.1796042838791436",
    "Delta_G": "23.20131262929499",
    "slack": "20.021708345415846",
    "normalized_gap": "0.0013479851772483812",
    "normalized_bound": "0.009836137684038947",
    "exact_optimal": "0",
    "bound_ok": "1",
    "cost_range": "2.0633106231689453",
    "R_D": "0.7138400253680608",
    "beta_eff": "840.5244456352688",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  },
  {
    "seed": "0",
    "step": "9",
    "M": "200",
    "F_dynamic": "-568.2067903137807",
    "F_exact": "-568.2067903137807",
    "F_min": "-2953.0849030751674",
    "objective_range": "2384.8781127613865",
    "gap": "0.0",
    "raw_gap": "0.0",
    "Delta_G": "132.8166201693639",
    "slack": "132.8166201693639",
    "normalized_gap": "0.0",
    "normalized_bound": "0.05569115648244978",
    "exact_optimal": "1",
    "bound_ok": "1",
    "cost_range": "2.448444366455078",
    "R_D": "0.6493801513881019",
    "beta_eff": "923.958007459686",
    "cost_degenerate": "0",
    "diversity_degenerate": "0",
    "objective_range_degenerate": "0"
  }
]
```
