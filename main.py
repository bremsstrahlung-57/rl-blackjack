# from gymnasium.utils.env_checker import check_env

from evaluation import evaluate, print_policy
from q_table import train
from rl_env import BlackJackRLEnviroment


def main():
    env = BlackJackRLEnviroment()
    # Q = train()
    # print(f"Trained {len(Q)} states")
    # evaluate(Q, env)
    # print_policy(Q)
    Q, N = train(num_episodes=2_000_000, epsilon_start=1.0, epsilon_min=1.0)

    for s in [(13,2,False), (12,5,False), (19,1,True), (18,5,True), (18,6,True)]:
        print(s, "Q:", Q[s], "N:", N[s])

    evaluate(Q, env)
    print_policy(Q)

if __name__ == "__main__":
    main()
