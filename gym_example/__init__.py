# import gymnasium as gym
# from .env import mazegame

# Register the environment
# gym.register(
#     id='MazeGame-v0',
#     entry_point='mazegame:MazeGameEnv', 
#     kwargs={'maze': None} 
# )

from gymnasium.envs.registration import register

register(
    id="Dynamic_env-v0",
    entry_point='gym_example.env:DynamicEnv',)

register(
    id="testenv-v0",
    entry_point='gym_example.env:test_envclass',)

register(
    id="stockenv-v0",
    entry_point='gym_example.env:Stock_env',)