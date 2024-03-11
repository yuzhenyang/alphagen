import json
import yaml
import fnmatch
import os
import sys
from typing import Optional, Tuple, Union, List
from enum import IntEnum
from datetime import datetime
import fire
import numpy as np
import pdb
import torch
import uuid

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <config.yml>")
    sys.exit(1)

config_yml = sys.argv[1]

with open(config_yml, 'r') as file:
    conf = yaml.safe_load(file)

def uuid_path(parent):
    return parent + "/" + str(uuid.uuid4()) + "/"

def format_date(insr):
    b, e = insr.split('-')
    if len(b) != 8 or len(e) > 8:
        raise f"Date range: {insr} format error"
    if len(e) < 8:
        e = b[:(8-len(e))] + e
    return (f"{b[0:4]}-{b[4:6]}-{b[6:8]}", f"{e[0:4]}-{e[4:6]}-{e[6:8]}")

seed, code, pool = conf['seed'], conf['code'], conf['pool']
gpuid = int(conf['gpu'])
traindr, verdr, testdr = format_date(conf['train']), format_date(conf['verify']), format_date(conf['test'])

step = conf['step'] if 'step' in conf else None
if step is None:
    step = min(1000_000, max(500_000, pool * pool * 10))

if isinstance(seed, int):
    seed = (seed, )

provider_uri, feature_pattern = conf['uri'], conf['features']
print("Provider URI: ", provider_uri)

print(seed, code, pool, step)
print(traindr, verdr, testdr)


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
# exec(feature_code)

feature_path = uuid_path('./features')
os.makedirs(feature_path, exist_ok = True)

with open(feature_path + "features.py", "w") as ff:
    ff.write(feature_code)

print(feature_path)
sys.path.insert(0, os.path.realpath(feature_path))

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
from features import FeatureType

from alphagen.data.expression import *

test = f"print(FeatureType.{feature_names[0].upper()})"
exec(test)


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
    device = torch.device(f'cuda:{gpuid}')

    #vwap = Feature(FeatureType.WDB_ASHAREENERGYINDEXADJ_BOLL_LOWER)
    vwap = Feature(FeatureType.MD_STD_VWP)
    target = Log(Ref(vwap, -6) / Ref(vwap, -1))
    # target = Feature(FeatureType.VWAP)

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockData(instrument=instruments,
                           start_time=traindr[0],
                           end_time=traindr[1],
                           device=device)
    data_valid = StockData(instrument=instruments,
                           start_time=verdr[0],
                           end_time=verdr[1],
                           device=device)
    data_test = StockData(instrument=instruments,
                          start_time=testdr[0],
                          end_time=testdr[1],
                          device=device)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device
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
