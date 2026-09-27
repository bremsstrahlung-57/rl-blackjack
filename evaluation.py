import numpy as np


def evaluate(Q, env, num_episodes=500_000):
    wins = losses = draws = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        wins += reward == 1
        losses += reward == -1
        draws += reward == 0
    total = wins + losses + draws
    print(f"Win: {wins/total:.3%}  Loss: {losses/total:.3%}  Push: {draws/total:.3%}")

def print_policy(Q):
    for usable_ace in [False, True]:
        print(f"\n{'Usable Ace' if usable_ace else 'No Usable Ace'}")
        header = "P\\D " + " ".join(f"{d:>3}" for d in range(1, 11))
        print(header)
        for player in range(21, 3, -1):
            row = []
            for dealer in range(1, 11):
                state = (player, dealer, usable_ace)
                if state in Q:
                    action = "H" if np.argmax(Q[state]) == 1 else "S"
                else:
                    action = "."
                row.append(f"{action:>3}")
            print(f"{player:>3} " + " ".join(row))
