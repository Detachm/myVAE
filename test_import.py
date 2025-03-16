import os
os.environ['MUJOCO_GL'] = 'cgl'

from SAR_tutorial_utils import *

print("Import successful!")

# 测试train函数
env_name = 'myoLegWalk-v0'
policy_name = 'play_period'
timesteps = 1.5e6
seed = '0'

train(env_name, policy_name, timesteps, seed) 