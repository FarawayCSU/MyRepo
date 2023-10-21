# 导入进程池
from multiprocessing import Pool
import os, time
from ray import tune
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from ray.rllib.algorithms.ars.ars import ARS, ARSConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray import air


env = BoptestGymEnv(
    url="http://127.0.0.1:5000/",
    actions=['oveHeaPumY_u'],
    observations={'time': (0, 604800),
                  'reaTZon_y': (280., 310.),
                  'TDryBul': (265, 303),
                  'HDirNor': (0, 862),
                  'InternalGainsRad[1]': (0, 219),
                  'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
                  'LowerSetp[1]': (280., 310.),
                  'UpperSetp[1]': (280., 310.)},
    scenario={'electricity_price': 'dynamic'},
    predictive_period=24 * 3600,
    regressive_period=6 * 3600,
    random_start_time=True,
    excluding_periods=[(16 * 24 * 3600, 30 * 24 * 3600), (108 * 24 * 3600, 122 * 24 * 3600)],
    max_episode_length=14 * 24 * 3600,
    warmup_period=24 * 3600,
    step_period=60 * 60)



class Job:
    def __init__(self, tuner, name: str):
        self.t = tuner
        self.name = name

    def run(self):
        self.t.fit()


def training(job: Job):
    print(f"{os.getpid()}号进程开始执行{job.name}任务")
    start = time.time()
    job.run()
    end = time.time()
    print(f"{job.name}任务执行完毕，耗时{end - start}秒")


if __name__ == '__main__':
    pool = Pool(3)
    jobs = list();
    # 定义分别的tune，在这下面

    env = BoptestGymEnv(
        url="http://127.0.0.1:5000/",
        actions=['oveHeaPumY_u'],
        observations={'time': (0, 604800),
                      'reaTZon_y': (280., 310.),
                      'TDryBul': (265, 303),
                      'HDirNor': (0, 862),
                      'InternalGainsRad[1]': (0, 219),
                      'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
                      'LowerSetp[1]': (280., 310.),
                      'UpperSetp[1]': (280., 310.)},
        scenario={'electricity_price': 'dynamic'},
        predictive_period=24 * 3600,
        regressive_period=6 * 3600,
        random_start_time=True,
        excluding_periods=[(16 * 24 * 3600, 30 * 24 * 3600), (108 * 24 * 3600, 122 * 24 * 3600)],
        max_episode_length=14 * 24 * 3600,
        warmup_period=24 * 3600,
        step_period=60 * 60)

    env1 = BoptestGymEnv(
        url="http://127.0.0.1:5001/",
        actions=['oveHeaPumY_u'],
        observations={'time': (0, 604800),
                      'reaTZon_y': (280., 310.),
                      'TDryBul': (265, 303),
                      'HDirNor': (0, 862),
                      'InternalGainsRad[1]': (0, 219),
                      'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
                      'LowerSetp[1]': (280., 310.),
                      'UpperSetp[1]': (280., 310.)},
        scenario={'electricity_price': 'dynamic'},
        predictive_period=24 * 3600,
        regressive_period=6 * 3600,
        random_start_time=True,
        excluding_periods=[(16 * 24 * 3600, 30 * 24 * 3600), (108 * 24 * 3600, 122 * 24 * 3600)],
        max_episode_length=14 * 24 * 3600,
        warmup_period=24 * 3600,
        step_period=60 * 60)

    tune0.register_env('BOPTEST-gym0', lambda config: BoptestGymEnv(env, **config))
    tune1.register_env('BOPTEST-gym1', lambda config: BoptestGymEnv(env1, **config))

    ddpg_config = DDPGConfig()
    ppo_config = PPOConfig()

    ddpg_config = ddpg_config.environment(env="BOPTEST-gym0")
    ppo_config = ppo_config.environment(env="BOPTEST-gym1")

    t1=tune1.Tuner("PPO",
               run_config=air.RunConfig(stop={"episode_reward_mean": -200}),
               param_space=ppo_config.to_dict())

    t2=tune0.Tuner("DDPG",
               run_config=air.RunConfig(stop={"episode_reward_mean": -200}),
               param_space=ddpg_config.to_dict())

    # 定义好了后，初始化Job，插入到jobs中
    # t = None  # 假设为t，名字是测试
    # j = Job(t, "测试")
    jobs.append(Job(t1,"一号"))
    jobs.append(Job(t2,"二号"))
    for i in jobs:
        pool.apply_async(training,args=(i,))

    pool.close()
    pool.join()
    print("所有任务已完成！")