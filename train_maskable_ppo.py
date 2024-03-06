import json
import yaml
import fnmatch
import os
from typing import Optional, Tuple, Union, List
from enum import IntEnum
from datetime import datetime
import fire
import numpy as np
import pdb
import torch

with open("config.yml", 'r') as file:
        conf = yaml.safe_load(file)

seed = conf['seed']
code = conf['code']
pool = conf['pool']
step = conf['step'] if 'step' in conf else None
if step is None:
    step = max(1000_000, min(500_000, pool * pool * 2))

if isinstance(seed, int):
    seed = (seed, )

provider_uri, feature_pattern = conf['uri'], conf['features']
print("Provider URI: ", provider_uri)

# get first sub directory
features_path = provider_uri + "/features"
contents = os.listdir(features_path)
first_subdirectory = None
for item in contents:
    item_path = os.path.join(features_path, item)
    if os.path.isdir(item_path):
        first_subdirectory = item_path
        break

# get feature names from sub directory
feature_names = []
files = os.listdir(first_subdirectory)
for file in files:
    for pattern in feature_pattern:
        if fnmatch.fnmatch(file, pattern):
            feature_names.append(file[:-8]) # remove ".day.bin"
            break

feature_code = "from enum import IntEnum\n\nclass FeatureType(IntEnum):\n"
for i, feature in enumerate(feature_names):
    feature_code += f"    {feature.upper()} = {i}\n"
exec(feature_code)
test = f"print(FeatureType.{feature_names[0].upper()})"
exec(test)

with open("alphagen_qlib/features.py", "w") as ff:
    ff.write(feature_code)

# start imports here for class FeatureType(IntEnum) ready
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_mask import MaskablePPO
from alphagen.data.calculator import AlphaCalculator

from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.features import FeatureType

from alphagen.data.expression import *


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 valid_calculator, # : AlphaCalculator,
                 test_calculator, #: AlphaCalculator,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        self.logger.record('test/ic', ic_test)
        self.logger.record('test/rank_ic', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self): # -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self): # -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
):
    reseed_everything(seed)

    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    #vwap = Feature(FeatureType.WDB_ASHAREENERGYINDEXADJ_BOLL_LOWER)
    vwap = Feature(FeatureType.MD_STD_VWP)
    target = Log(Ref(vwap, -6) / Ref(vwap, -1))
    # target = Feature(FeatureType.VWAP)

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockData(instrument=instruments,
                           start_time='2011-01-01',
                           end_time='2018-12-31')
    data_valid = StockData(instrument=instruments,
                           start_time='2019-01-01',
                           end_time='2019-12-31')
    data_test = StockData(instrument=instruments,
                          start_time='2020-01-01',
                          end_time='2020-11-15')
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"new_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='log/checkpoints',
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log='log/tb/log',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


def fire_helper(
    seed: Union[int, Tuple[int]],
    code: str,
    pool: int,
    step: int = None
):
    if isinstance(seed, int):
        seed = (seed, )
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000,
        200: 400_000,
        500: 500_000,
        1000: 800_000
    }
    for _seed in seed:
        main(_seed,
             code,
             pool,
             default_steps[int(pool)] if step is None else int(step)
             )

for _seed in seed:
    main(_seed, code, pool, step)