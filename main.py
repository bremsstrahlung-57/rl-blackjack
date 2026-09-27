# from gymnasium.utils.env_checker import check_env

from evaluation import evaluate, print_policy
from q_table import train
from rl_env import BlackJackRLEnviroment


def main():
    env = BlackJackRLEnviroment()
    Q = train()
    print(f"Trained {len(Q)} states")
    evaluate(Q, env)
    print_policy(Q)

if __name__ == "__main__":
    main()
