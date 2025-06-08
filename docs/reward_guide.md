# Reward Function Design for Rocket League Bots

Designing an effective reward function is critical for training a bot that behaves as intended. The metrics you choose should encourage the bot to pursue useful objectives while avoiding unintended behaviors.

## Competitive Play

- **Scoring Goals**: Reward the bot when it scores or increases the likelihood of scoring. You can measure this by tracking ball proximity to the opponent goal or by checking `GameTickPacket.game_info.is_goal_scored`.
- **Defending**: Provide positive rewards for clearing the ball from the defensive half, blocking shots, or saving goals.
- **Boost Usage**: Encourage efficient boost collection and spending. Use `GameTickPacket.game_cars[car_index].boost` to evaluate boost levels. Penalize unnecessary boost consumption or reward maintaining sufficient boost for plays.

## Freestyle and Creative Play

- **Aerial Control**: Reward the bot for performing aerial maneuvers and hitting the ball while airborne. Metrics such as the car's `physics.location.z` and `physics.velocity` can help detect aerials.
- **Chain Combos**: Provide incremental rewards for chaining multiple hits without the ball touching the ground. Track touch events via `GameTickPacket.game_ball.latest_touch`.
- **Style Points**: Creative play often values flair. Add small bonuses for spins, flips, or other stylish movements detected through `GameTickPacket.game_cars[car_index].physics.rotation`.

## Balancing Rewards

- **Positive vs. Negative**: Combine bonuses for desirable actions with penalties for mistakes (e.g., own goals or missed touches). Keep the magnitude of penalties similar to rewards so the bot learns balanced behavior.
- **Reward Shaping**: Start with simple sparse rewards (like goals) and add shaping terms (ball velocity toward the net, positioning) to speed up learning without overwhelming the main objective.

## Example Packet Metrics

`GameTickPacket` contains many values that are useful for rewards:

- `game_info.seconds_elapsed` – Track time for timed bonuses or penalties.
- `game_ball.physics.location` – Determine ball position relative to goals.
- `game_cars[car_index].physics.location` – Car positioning for defense or freestyling.
- `game_cars[car_index].boost` – Current boost amount.
- `game_ball.latest_touch` – Who last hit the ball and when.

Mixing these metrics allows you to craft rewards tailored to your training goals.
