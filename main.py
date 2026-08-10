from rl_env import BlackJackRLEnviroment
from gymnasium.utils.env_checker import check_env


def main():
    env = BlackJackRLEnviroment()
    check_env(env)

if __name__ == "__main__":
    main()
