"""Tiny chain MDP exercise for the IOAI 2027 RL extension; no gym dependency."""
import argparse
import numpy as np
from .blocks.rl import q_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    rng, q = np.random.default_rng(7), np.zeros((7, 2))
    for _ in range(args.episodes):
        state = 0
        for _ in range(50):
            action = int(rng.integers(2)) if rng.random() < 0.25 else int(q[state].argmax())
            nxt = int(np.clip(state + (2 * action - 1), 0, 6))
            done = nxt == 6
            q_update(q, state, action, 1.0 if done else -0.01, nxt, done)
            state = nxt
            if done:
                break
    print({"fixture": "chain_mdp", "policy_0_to_5": q[:6].argmax(1).tolist(), "right_action": 1})


if __name__ == "__main__":
    main()
