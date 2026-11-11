# Code Review: learned-mppi

Reviewed as a staff engineer would review a research-code pull request that compares learned dynamics models under sampling MPC. Scope: `lmppi/` (plant, data, models, MPPI), the experiment scripts, tests and build files.

## 1. Overall assessment

Sound research code with a clear question, paired seeds, a held-out test set, and a README that reports negative results (the Koopman model, the hybrid residual) as carefully as positive ones. The plant and the torch dynamics agree to float64 precision, and the cart-pole equations, including the pivot-friction coupling, check out by hand derivation.

The review found no correctness defect in the algorithms. It found self-documentation gaps (unnamed cost weights and thresholds, a library module that changed PyTorch's global default dtype on import) and one statistical gap in how the closed-loop results were presented. The code changes made in this pass are behavior-preserving, and a rerun of two control episodes from the saved models reproduced the committed numbers to every printed digit.

## 2. Scorecard

| Area | Score | Basis |
|---|---|---|
| Correctness | 8 | Dynamics derived and tested against an independent integrator, angle wrapping tested, MPPI balances with the true model in a test. |
| Architecture | 7 | One `DynModel` interface for seven models keeps the comparison fair. `models.py` holds all seven in 426 lines and would be easier to navigate split by family. |
| Readability | 7 | Clear module docstrings. Math-style short names (`th`, `w`, `xb`) are dense outside the dynamics functions. |
| Self-documentation | 7 | After this pass, cost weights, thresholds and plant constants are named with units. Model hyperparameters are still constructor defaults without reasons. |
| Maintainability | 7 | Staged pipeline with cached data and models. Results depend on pickled model objects, so a refactor of a model class can break old pickles. |
| Testing | 7 | Eight focused tests including an end-to-end pipeline smoke test. No test pins a closed-loop number, so a silent behavior change would only show up in a full rerun. |
| Performance | Not assessed | Python research code. Step times are reported, and the README says which models could run at 50 Hz. |
| Reliability | 7 | Diverged rollouts are mapped to a large finite cost so they cannot poison the softmax. Seeds are explicit throughout. |
| API design | 7 | `step(x, u, aux, member)` is uniform across models. `member="split"` is a stringly typed mode. |
| Domain-specific correctness | 8 | ZOH at 50 Hz over RK4 at 1 kHz, exact discretization of the actuator lag, saturation and deadzone modelled on both sides, history features long enough to cover the hidden lag. |
| Safety and robustness | Not assessed | Simulation-only benchmark with no safety function. |

## 3. Problems

### Fixed in this review

| Severity | Location | What | Why | Fix |
|---|---|---|---|---|
| Medium | `lmppi/models.py` | `torch.set_default_dtype(torch.float32)` at import time. | A library import silently overrode a global setting for any caller. It was a no-op only because float32 is already the default. | Removed. |
| Medium | `lmppi/mppi.py` | Stage-cost weights, the terminal multiplier, the 2 m track limit, the 12° tolerance and the 2 s success window were bare literals, and `0.02` stood for the sample period in two places. | The cost is the controller's specification and had no units or reasons next to it. | `CostWeights` (frozen dataclass, units in comments), `TRACK_HALF_LENGTH_M`, `UPRIGHT_TOLERANCE_DEG`, `SUCCESS_WINDOW_S`, `DT`. |
| Low | `plant.py`, `models.py` | The Coulomb smoothing width (0.02 m/s) appeared separately in the NumPy plant and the torch model. | Changing one would have made the "oracle" model disagree with the plant without any test failing until the tolerance was hit. | One constant, `COULOMB_SMOOTHING_M_S`, imported by both. |
| Low | README | "7/10 versus 10/10 is suggestive" without numbers. | A reader cannot tell which differences are real. | Fisher's exact test added: p = 0.21 for 7/10 against 10/10, p = 0.03 for 5/10 against 10/10. |

### Open

| Severity | Location | What | Why | Recommended fix |
|---|---|---|---|---|
| Medium | experiment design | 10 episodes per model. | Only the large gaps (Koopman, ensemble round 1 versus round 2) are statistically clear. | 30 or more seeds for the models whose ranking matters, and report Wilson intervals. |
| Low | `EnsembleModel.fit` | The validation set selects the checkpoint and is also the reported validation MSE. | The reported number is optimistically biased. The test-set prediction errors in the README are unaffected. | Report test-set MSE beside it, or label it as the selection metric. |
| Low | `scripts/` | Results are pickled model objects. | A class change can make old `models.pkl` files unloadable. | Save `state_dict`s and constructor arguments instead. |
| Low | `MPPI` | The unperturbed plan is always sample 0, so with ensemble splitting it is only ever scored on member 0. | Slight bias toward member 0's opinion of the current plan. | Evaluate the nominal plan on every member, or rotate the member assignment per step. |

## 4. Readability and self-documentation

```python
# before
cost = cost + 5.0 * stage_cost(xb, torch.zeros_like(cost))
last = xs[-100:]  # last 2 s
success = bool(np.all(np.abs(wrap(last[:, 1] - np.pi)) < np.radians(12)) and np.all(np.abs(last[:, 0]) < 2.0))

# after
cost = cost + COST.terminal_scale * stage_cost(xb, torch.zeros_like(cost))
last = xs[-round(SUCCESS_WINDOW_S / DT):]
success = bool(np.all(np.abs(wrap(last[:, 1] - np.pi)) < tolerance) and np.all(np.abs(last[:, 0]) < TRACK_HALF_LENGTH_M))
```

The plant's docstring is a model of what the rest should look like: it says what differs from the textbook, by how much, and what is hidden from the learner.

## 5. Architecture and design

The `DynModel` interface (`init_aux`, `step`, `rollout`, `fit`) is the right abstraction: every model is scored by the same rollout code and the same MPPI, so differences come from the models. Ensemble member splitting inside `step` rather than inside MPPI keeps the controller model-agnostic. The staged script (data, fit, predict, control, round 2) with cached artifacts makes the 35-minute pipeline restartable.

## 6. Testing and verification

| Claim | Verified by |
|---|---|
| Torch dynamics equal the NumPy plant | Simulation test, float64 |
| Cart-pole equations, pivot friction coupling | Math: derived from the Lagrangian during this review |
| Upright is an equilibrium | Test |
| Smooth deadzone matches the hard one away from the corner and is monotone | Test |
| RFF features approximate the RBF kernel | Test |
| Grey-box gradients reduce the loss | Test |
| MPPI balances with the true model | Closed-loop test |
| The refactor changed nothing | Two control episodes rerun from the saved models matched the committed CSV |
| Any of it on hardware | Not applicable, the plant is a model |

## 7. Prioritized action plan

Must fix before commit: none outstanding.

Before sharing as a benchmark: more seeds with confidence intervals, and test-set rather than validation-set model error in every table.

Recommended: save models as state dicts, and split `models.py` by model family.

Optional: score the nominal plan on every ensemble member.

## 8. Final verdict

| Context | Verdict |
|---|---|
| Personal or portfolio project | STRONG |
| Shared codebase | ACCEPTABLE |
| Simulation study | STRONG for the qualitative conclusions, ACCEPTABLE for the close rankings until more seeds are run |
| Hardware | Not applicable |
