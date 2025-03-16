import os
os.environ['MUJOCO_GL'] = 'glfw'

from myosuite.utils import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class SaveSuccesses(BaseCallback):
    """
    sb3 callback used to calculate and monitor success statistics.
    """
    def __init__(self, check_freq: int, log_dir: str, env_name: str, verbose: int = 1):
        super(SaveSuccesses, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'ignore')
        self.check_for_success = []
        self.success_buffer = []
        self.success_results = []
        self.env_name = env_name

    def _on_rollout_start(self) -> None:
        self.check_for_success = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self) -> None:
        if sum(self.check_for_success) > 0:
            self.success_buffer.append(1)
        else:
            self.success_buffer.append(0)
        if len(self.success_buffer) > 0:
            self.success_results.append(sum(self.success_buffer)/len(self.success_buffer))

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.check_for_success.append(self.locals['infos'][0].get('solved', False))
        return True

def train(env_name, policy_name, timesteps, seed):
    """
    Trains a policy using sb3 implementation of SAC.
    """
    env = gym.make(env_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))
    
    model = SAC('MlpPolicy', env, learning_rate=linear_schedule(.001), buffer_size=int(3e5),
            learning_starts=1000, batch_size=256, tau=.02, gamma=.98, train_freq=(1, "episode"),
            gradient_steps=-1, policy_kwargs=policy_kwargs, verbose=1)
    
    succ_callback = SaveSuccesses(check_freq=1, env_name=env_name+'_'+seed, 
                             log_dir=f'{policy_name}_successes_{env_name}_{seed}')
    
    model.set_logger(configure(f'{policy_name}_results_{env_name}_{seed}'))
    model.learn(total_timesteps=int(timesteps), callback=succ_callback, log_interval=4)
    model.save(f"{policy_name}_model_{env_name}_{seed}")
    env.save(f'{policy_name}_env_{env_name}_{seed}')

print("开始训练过程...")
train('myoLegWalk-v0', 'play_period', 1.5e6, '0') 