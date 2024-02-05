from gymnasium.envs.registration import register

register(
    id="grid/GridWorld-v0",
    entry_point="grid.envs:GridWorldEnv",
)
