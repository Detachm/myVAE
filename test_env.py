import os; os.environ['MUJOCO_GL'] = 'glfw'; import gymnasium as gym; import myosuite.envs; env = gym.make('myoLegWalk-v0'); print('Environment created successfully!')
