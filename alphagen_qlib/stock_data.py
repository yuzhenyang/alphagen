from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch
import pdb

# class FeatureType(IntEnum):
    # VWAP = 0
    # WDB_ASHAREINTENSITYTRENDADJ_DMI_ADX = VWAP + 1
    # WDB_ASHAREINTENSITYTRENDADJ_MA_60D = VWAP + 2
    # WDB_ASHAREINTENSITYTRENDADJ_BOTTOMING_B = VWAP + 3
    # WDB_ASHAREINTENSITYTRENDADJ_PRICEOSC = VWAP + 4
    # WDB_ASHAREINTENSITYTRENDADJ_DMI_MDI = VWAP + 5
    # WDB_ASHAREINTENSITYTRENDADJ_DMA_DDD = VWAP + 6
    # WDB_ASHAREINTENSITYTRENDADJ_TRMA = VWAP + 7
    # WDB_ASHAREINTENSITYTRENDADJ_DMI_ADXR = VWAP + 8
    # WDB_ASHAREINTENSITYTRENDADJ_MA_120D = VWAP + 9
    # WDB_ASHAREINTENSITYTRENDADJ_MA_250D = VWAP + 10
    # WDB_ASHAREINTENSITYTRENDADJ_MA_10D = VWAP + 11
    # WDB_ASHAREINTENSITYTRENDADJ_SAR = VWAP + 12
    # WDB_ASHAREINTENSITYTRENDADJ_MA_20D = VWAP + 13
    # WDB_ASHAREINTENSITYTRENDADJ_MA_5D = VWAP + 14
    # WDB_ASHAREINTENSITYTRENDADJ_STRENGTH = VWAP + 15
    # WDB_ASHAREINTENSITYTRENDADJ_MTM = VWAP + 16
    # WDB_ASHAREINTENSITYTRENDADJ_MACD_DEA = VWAP + 17
    # WDB_ASHAREINTENSITYTRENDADJ_DMA_AMA = VWAP + 18
    # WDB_ASHAREINTENSITYTRENDADJ_BBI = VWAP + 19
    # WDB_ASHAREINTENSITYTRENDADJ_MACD_MACD = VWAP + 20
    # WDB_ASHAREINTENSITYTRENDADJ_BOTTOMING_D = VWAP + 21
    # WDB_ASHAREINTENSITYTRENDADJ_MTM_MTMMA = VWAP + 22
    # WDB_ASHAREINTENSITYTRENDADJ_MARKET = VWAP + 23
    # WDB_ASHAREINTENSITYTRENDADJ_MA_30D = VWAP + 24
    # WDB_ASHAREINTENSITYTRENDADJ_MACD_DIFF = VWAP + 25
    # WDB_ASHAREINTENSITYTRENDADJ_DMI_PDI = VWAP + 26
    # WDB_ASHAREINTENSITYTRENDADJ_EXPMA = VWAP + 27
    # WDB_ASHAREINTENSITYTRENDADJ_TRIX = VWAP + 28
    # WDB_ASHAREINTENSITYTRENDADJ_WEAKKNESS = VWAP + 29


# class FeatureType(IntEnum):
    # WAP = 0
    # WDB_ASHARESWINGREVERSETRENDADJ_KDJ_K = VWAP + 1
    # WDB_ASHARESWINGREVERSETRENDADJ_ADTM_ADTM = VWAP + 2
    # WDB_ASHARESWINGREVERSETRENDADJ_DPO_MADPO = VWAP + 3
    # WDB_ASHARESWINGREVERSETRENDADJ_CVLT = VWAP + 4
    # WDB_ASHARESWINGREVERSETRENDADJ_RC_50D = VWAP + 5
    # WDB_ASHARESWINGREVERSETRENDADJ_WR = VWAP + 6
    # WDB_ASHARESWINGREVERSETRENDADJ_CCI = VWAP + 7
    # WDB_ASHARESWINGREVERSETRENDADJ_BIAS36 = VWAP + 8
    # WDB_ASHARESWINGREVERSETRENDADJ_SLOWKD_D = VWAP + 9
    # WDB_ASHARESWINGREVERSETRENDADJ_BIAS = VWAP + 10
    # WDB_ASHARESWINGREVERSETRENDADJ_VHF = VWAP + 11
    # WDB_ASHARESWINGREVERSETRENDADJ_MI_MI12D = VWAP + 12
    # WDB_ASHARESWINGREVERSETRENDADJ_MASS = VWAP + 13
    # WDB_ASHARESWINGREVERSETRENDADJ_ATR_ATR14D = VWAP + 14
    # WDB_ASHARESWINGREVERSETRENDADJ_MI_A12D = VWAP + 15
    # WDB_ASHARESWINGREVERSETRENDADJ_KDJ_J = VWAP + 16
    # WDB_ASHARESWINGREVERSETRENDADJ_ADTM_ADTMMA = VWAP + 17
    # WDB_ASHARESWINGREVERSETRENDADJ_ROC_ROCMA = VWAP + 18
    # WDB_ASHARESWINGREVERSETRENDADJ_DPO = VWAP + 19
    # WDB_ASHARESWINGREVERSETRENDADJ_SI = VWAP + 20
    # WDB_ASHARESWINGREVERSETRENDADJ_RSI = VWAP + 21
    # WDB_ASHARESWINGREVERSETRENDADJ_ATR_TR14D = VWAP + 22
    # WDB_ASHARESWINGREVERSETRENDADJ_KDJ_D = VWAP + 23
    # WDB_ASHARESWINGREVERSETRENDADJ_ROC = VWAP + 24
    # WDB_ASHARESWINGREVERSETRENDADJ_BIAS612 = VWAP + 25
    # WDB_ASHARESWINGREVERSETRENDADJ_SRMI_9D = VWAP + 26
    # WDB_ASHARESWINGREVERSETRENDADJ_SLOWKD_K = VWAP + 27


class FeatureType(IntEnum):
    VWAP = 0
    WDB_ASHAREENERGYINDEXADJ_CDP_AL = VWAP + 1
    WDB_ASHAREENERGYINDEXADJ_PSY = VWAP + 2
    WDB_ASHAREENERGYINDEXADJ_OBV = VWAP + 3
    WDB_ASHAREENERGYINDEXADJ_BBIBOLL_BBI = VWAP + 4
    WDB_ASHAREENERGYINDEXADJ_MIKE_WR = VWAP + 5
    WDB_ASHAREENERGYINDEXADJ_MIKE_MS = VWAP + 6
    WDB_ASHAREENERGYINDEXADJ_MIKE_MR = VWAP + 7
    WDB_ASHAREENERGYINDEXADJ_PVT = VWAP + 8
    WDB_ASHAREENERGYINDEXADJ_MIKE_SS = VWAP + 9
    WDB_ASHAREENERGYINDEXADJ_BBIBOLL_UPR = VWAP + 10
    WDB_ASHAREENERGYINDEXADJ_ARBR_AR = VWAP + 11
    WDB_ASHAREENERGYINDEXADJ_MIKE_WS = VWAP + 12
    WDB_ASHAREENERGYINDEXADJ_CDP_AH = VWAP + 13
    WDB_ASHAREENERGYINDEXADJ_BOLL_MID = VWAP + 14
    WDB_ASHAREENERGYINDEXADJ_CR = VWAP + 15
    WDB_ASHAREENERGYINDEXADJ_PSYMA = VWAP + 16
    WDB_ASHAREENERGYINDEXADJ_WAD = VWAP + 17
    WDB_ASHAREENERGYINDEXADJ_WVAD_WVAD = VWAP + 18
    WDB_ASHAREENERGYINDEXADJ_OBV_OBV = VWAP + 19
    WDB_ASHAREENERGYINDEXADJ_CDP = VWAP + 20
    WDB_ASHAREENERGYINDEXADJ_BOLL_UPPER = VWAP + 21
    WDB_ASHAREENERGYINDEXADJ_BBIBOLL_DWN = VWAP + 22
    WDB_ASHAREENERGYINDEXADJ_WVAD_MAWVAD = VWAP + 23
    WDB_ASHAREENERGYINDEXADJ_ENV_LOWER = VWAP + 24
    WDB_ASHAREENERGYINDEXADJ_BOLL_LOWER = VWAP + 25
    WDB_ASHAREENERGYINDEXADJ_ENV_UPPER = VWAP + 26
    WDB_ASHAREENERGYINDEXADJ_MAWAD = VWAP + 27
    WDB_ASHAREENERGYINDEXADJ_MIKE_SR = VWAP + 28
    WDB_ASHAREENERGYINDEXADJ_CDP_NH = VWAP + 29
    WDB_ASHAREENERGYINDEXADJ_CDP_NL = VWAP + 30
    WDB_ASHAREENERGYINDEXADJ_ARBR_BR = VWAP + 31


class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0')) -> None:
        self._init_qlib()

        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_qlib(cls) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri="~/.qlib/qlib_data/cne", region=REG_CN)
        # qlib.init(provider_uri="~/.qlib/qlib_data/ZZ800_data", region=REG_CN)
        # qlib.init(provider_uri="~/.qlib/qlib_data/CSI300_data", region=REG_CN)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        real_end_time = cal[end_index + self.max_future_days]
        return (QlibDataLoader(config=exprs)  # type: ignore
                .load(self._instrument, real_start_time, real_end_time))

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df = df.stack().unstack(level=1)
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
