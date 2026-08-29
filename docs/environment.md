# Environment contract

This document describes environment ID `gravity-lab-sandbox-v1`, not the vendored classic game.
The classic executable currently has no supported step/reset observation API. Do not label sandbox
training runs as classic-physics runs.

## Markov decision process

At time `t`, an agent receives state observation `s_t`, selects action `a_t`, and receives reward
`r_(t+1)` and the next observation `s_(t+1)`. A transition is
`(s_t, a_t, r_(t+1), s_(t+1), terminated, truncated)`. An episode ends after a crash or finish
(`terminated`) or the configured step limit (`truncated`). Algorithms may bootstrap through a time
limit, but must not bootstrap through a terminal crash or finish.

The return is the discounted future reward

```text
G_t = r_(t+1) + gamma r_(t+2) + gamma^2 r_(t+3) + ...
```

where `gamma` in `[0, 1]` trades immediate reward against later reward. A policy maps observations
to actions. Evaluation uses a greedy/deterministic policy; exploration belongs only in training.

## Actions

| Value | Action |
|---:|---|
| 0 | coast |
| 1 | throttle |
| 2 | brake |
| 3 | lean back |
| 4 | lean forward |
| 5 | throttle + lean back |
| 6 | throttle + lean forward |
| 7 | brake + lean back |
| 8 | brake + lean forward |

Every environment call repeats the chosen action for `frame_skip` fixed integrations of
`time_step` seconds. An episode step counts one agent action, not one integration.

## Observation

The 12 `double` values are stable API order:

1. course progress relative to start and finish
2. bike height above local terrain, divided by 5
3. horizontal velocity divided by 14
4. vertical velocity divided by 10
5. sine of bike angle
6. cosine of bike angle
7. angular velocity divided by 8
8. local terrain slope divided by pi/2
9. terrain-height delta 1 m ahead, divided by 5
10. terrain-height delta 3 m ahead, divided by 5
11. terrain-height delta 6 m ahead, divided by 5
12. grounded flag (0 or 1)

This engineered vector is appropriate for small neural networks. Tabular Q-learning in the example
coarsens only a subset and therefore loses information; results must not be presented as though the
tabular and continuous observation spaces are equivalent. A later pixel/grid mode should be added
as a separate observation version, not by silently changing this vector.

## Rewards and endings

Each action receives `0.1 * horizontal_progress - 0.001`, plus `+10` on reaching the finish or `-5`
on crashing. The finish is crossing the selected map's finish x-coordinate. A crash occurs on a
hard landing, an excessive bike/terrain angle at contact, rider-head contact, or leaving the left
terrain edge. Reaching `max_episode_steps` is truncation, not failure.

Reward constants and observation scaling are part of environment version v1. Changes require a
new version and must not be mixed in one comparison.

## Q-learning baseline

For learning rate `alpha`, discount `gamma`, and next-state greedy value, tabular Q-learning uses:

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

The bracketed term is the temporal-difference error. For a true terminal transition, the
`gamma max Q` term is zero. The included example uses epsilon-greedy exploration with a seeded RNG
and an exponentially decreasing epsilon. It is a teaching baseline, not a reported benchmark.

## API ownership

`Map` owns validation/interpolation. `Environment` owns simulation and episode semantics. Apps own
rendering and controls. Agents own exploration, networks, replay, optimization, checkpointing, and
evaluation. Nothing in the core imports an RL or graphics framework.
