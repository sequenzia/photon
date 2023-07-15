from __future__ import annotations
from photon.photon import Photon
from photon.gamma import Gamma
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import tensorflow as tf

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from photon import Photon
    from photon.gamma import Gamma


class Networks():

    def __init__(self,
                 photon_id: str,
                 data_dir: str,
                 data_fn: str,
                 data_res: int,
                 data_cols: str,
                 x_groups_on: bool,
                 dirs_on: str,
                 diag_on: bool,
                 msgs_on: bool,
                 name: str = '',
                 photon: Photon = None,
                 float_x: int = 32):

        self.is_built = False

        self.photon_load_id = photon_id

        if name is None:
            self.name = 'Photon Network'
        else:
            self.name = name

        if photon is None:
            self.photon: Photon = Photon()
        else:
            self.photon: Photon = photon

        self.photon.setup_photon(self.photon_load_id)

        self.framework = photon.framework

        self.gamma: Gamma = Gamma(self)

        self.x_groups_on = x_groups_on

        self.dirs_on = dirs_on

        self.data = None
        self.data_dir = data_dir
        self.data_fn = data_fn
        self.data_res = data_res
        self.data_cols = data_cols

        self.diag_on = diag_on
        self.msgs_on = msgs_on

        self.msgs_on['epochs'] = False

        if float_x == 16:
            self.float_x = 'float16'
            self.dtype = np.float16
            self.framework_dtype = torch.float16 if self.framework == 'torch' else tf.float16

        if float_x == 32:
            self.float_x = 'float32'
            self.dtype = np.float32
            self.framework_dtype = torch.float32 if self.framework == 'torch' else tf.float32

        if float_x == 64:
            self.float_x = 'float64'
            self.dtype = np.float64
            self.framework_dtype = torch.float64 if self.framework == 'torch' else tf.float64

        # -- load data -- #
        self.load_data()

        self.n_trees = 0
        self.n_branches = 0
        self.n_chains = 0
        self.n_runs = 0

        self.trees = []
        self.branches = []
        self.chains = []
        self.runs = []

    def add_tree(self, tree):

        self.trees.append(tree)
        self.n_trees += 1

        return self.n_trees - 1, self.data

    def add_branch(self, branch):

        self.branches.append(branch)
        self.n_branches += 1

        return self.n_branches - 1

    def load_data(self):

        self.data_fp = self.data_dir + '/' + self.data_fn + '.parquet'

        self.data = self.Data(self.framework, self.framework_dtype)

        self.setup_cols(self.data_cols)

        all_cols = self.data.all_cols + ['is_close']

        self.data.full_bars = pq.read_table(self.data_fp).to_pandas().astype(self.dtype)[all_cols]

        return

    def setup_cols(self, data_cols):

        # ------- x_cols ------- #

        self.data.x_cols = list(data_cols['x_cols'].keys())

        # ------- c_cols ------- #

        if data_cols['c_cols'] is not None:
            self.data.c_cols = list(data_cols['c_cols'].keys())

        if data_cols['c_cols'] is None:
            self.data.c_cols = []

        # ------- y_cols ------- #

        if data_cols['y_cols'] is not None:
            self.data.y_cols = list(data_cols['y_cols'].keys())

        if data_cols['y_cols'] is None:
            self.data.y_cols = []

        # ------- t_cols ------- #

        if data_cols['t_cols'] is not None:
            self.data.t_cols = list(data_cols['t_cols'].keys())

        if data_cols['t_cols'] is None:
            self.data.t_cols = []

        self.data.f_cols = data_cols['f_cols']

        self.data.n_x_cols = len(self.data.x_cols)
        self.data.n_c_cols = len(self.data.c_cols)
        self.data.n_y_cols = len(self.data.y_cols)
        self.data.n_t_cols = len(self.data.t_cols)

        # -- x cols -- #
        for i in range(self.data.n_x_cols):

            _col = self.data.x_cols[i]
            _base = data_cols['x_cols'][_col]

            self.data.agg_data[_col] = _base['seq_agg']

            if _base['ofs_on']:
                self.data.ofs_data['x'].append(_col)

            if not self.x_groups_on:
                if _base['nor_on']:
                    self.data.nor_data['x'].append(_col)

        # -- c cols -- #
        if data_cols['c_cols'] is not None:

            for i in range(self.data.n_c_cols):

                _col = self.data.c_cols[i]
                _base = data_cols['c_cols'][_col]

                self.data.agg_data[_col] = _base['seq_agg']

                if _base['ofs_on']:
                    self.data.ofs_data['c'].append(_col)

                if _base['nor_on']:
                    self.data.nor_data['c'].append(_col)

        # -- y cols -- #
        if data_cols['y_cols'] is not None:

            for i in range(self.data.n_y_cols):

                _col = self.data.y_cols[i]
                _base = data_cols['y_cols'][_col]

                self.data.agg_data[_col] = _base['seq_agg']

                if _base['ofs_on']:
                    self.data.ofs_data['y'].append(_col)

                if _base['nor_on']:
                    self.data.nor_data['y'].append(_col)

        # -- t cols -- #
        for i in range(self.data.n_t_cols):

            _col = self.data.t_cols[i]
            _base = data_cols['t_cols'][_col]

            self.data.agg_data[_col] = _base['seq_agg']

            if _base['ofs_on']:
                self.data.ofs_data['t'].append(_col)

            if _base['nor_on']:
                self.data.nor_data['t'].append(_col)

        if self.x_groups_on:
            self.data.x_groups = self.setup_x_groups(data_cols['x_cols'])

        self.data.close_cols = ['bar_idx',
                                'day_idx',
                                'BAR_TP']

        self.data.all_cols = self.data.x_cols + \
            self.data.c_cols + self.data.y_cols + self.data.t_cols

        x_ed = self.data.n_x_cols
        c_st = x_ed
        c_ed = c_st + self.data.n_c_cols
        y_st = c_ed
        y_ed = y_st + self.data.n_y_cols
        t_st = y_ed

        self.data.slice_configs = {'x_slice': np.s_[..., :x_ed],
                                   'c_slice': np.s_[..., c_st:c_ed],
                                   'y_slice': np.s_[..., y_st:y_ed],
                                   't_slice': np.s_[..., t_st:]}

    def setup_x_groups(self, x_cols):

        pr_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'pr':
                    pr_group.append(k)

        vol_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'vol':
                    vol_group.append(k)

        atr_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'atr':
                    atr_group.append(k)

        roc_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'roc':
                    roc_group.append(k)

        zsc_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'zsc':
                    zsc_group.append(k)

        return {'pr_group': pr_group,
                'vol_group': vol_group,
                'atr_group': atr_group,
                'roc_group': roc_group,
                'zsc_group': zsc_group}

    @dataclass
    class Data:

        framework: str
        dtype: Union[tf.DType, torch.dtype]

        x_cols: List = field(default_factory=lambda: [[]])
        c_cols: List = field(default_factory=lambda: [[]])
        y_cols: List = field(default_factory=lambda: [[]])
        t_cols: List = field(default_factory=lambda: [[]])

        close_cols: List = field
        all_cols: List = field(default_factory=lambda: [[]])

        n_x_cols: List = 0
        n_c_cols: List = 0
        n_y_cols: List = 0
        n_t_cols: List = 0

        agg_data: Dict = field(default_factory=lambda: {})

        ofs_data: List = field(default_factory=lambda: {
            'x': [],
            'c': [],
            'y': [],
            't': []})

        nor_data: List = field(default_factory=lambda: {
            'x': [],
            'c': [],
            'y': [],
            't': []})

        rank: int = 2

        seq_on: bool = False

        full_bars: pd.DataFrame = field(init=False)
