from gymnasium.utils.env_checker import check_env

from rl_env import BlackJackRLEnviroment


def main():
    env = BlackJackRLEnviroment()
    check_env(env)

if __name__ == "__main__":
    main()
