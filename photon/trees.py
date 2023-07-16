from typing import List, Dict, Optional, Union
import numpy as np
import tensorflow as tf
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, TensorDataset

from photon import Photon, Networks


class Trees():

    def __init__(self,
                 batch_size: int,
                 train_days: int,
                 test_days: int,
                 val_days: int,
                 seq_days: int,
                 seq_len: int,
                 seq_agg: int,
                 test_on: bool,
                 val_on: bool,
                 masking: Dict,
                 shuffle: Dict,
                 preproc: Dict,
                 outputs_on: bool,
                 seed: int,
                 save_raw: bool = True,
                 name: str = '',
                 samples_pd: int = None,
                 photon: Photon = None,
                 network: Networks = Networks,
                 network_config: Dict = {}):

        self.is_built = False

        if network is None:
            self.network = Networks(photon=photon,
                                    **network_config)
        else:
            self.network = network

        if name is None:
            self.name = 'Photon Tree'
        else:
            self.name = name

        self.val_on = val_on
        self.test_on = test_on

        self.n_branches = 0
        self.n_chains = 0

        self.branches = []
        self.chains = []

        self.tree_idx, self.data = self.network.add_tree(self)

        self.data.seq_days = seq_days
        self.data.seq_len = seq_len
        self.data.seq_agg = seq_agg
        self.data.seq_depth = seq_len
        self.data.res = self.network.data_res

        self.data.train_days = train_days
        self.data.test_days = test_days
        self.data.val_days = val_days

        self.data.masking = masking

        self.data.shuffle = shuffle
        self.data.preproc = preproc
        self.data.outputs_on = outputs_on
        self.data.seed = seed
        self.data.save_raw = save_raw

        self.data.batch_size = batch_size

        if samples_pd is None:
            self.data.samples_pd = int(23400 / self.data.res)
        else:
            self.data.samples_pd = samples_pd

        if self.data.seq_days:
            self.data.seq_on = True

        self.datasets = {'train': None,
                         'val': None,
                         'test': None}

    def load_data(self):

        self.data.preproc_trans = {
            'train': {'x_cols': None,
                      'c_cols': None,
                      'y_cols': None,
                      't_cols': None,
                      'pr_cols': None,
                      'vol_cols': None,
                      'atr_cols': None,
                      'roc_cols': None,
                      'zsc_cols': None},
            'test': {'y_cols': None,
                     't_cols': None},
            'val': {'y_cols': None,
                    't_cols': None}}

        self.data.store = self.create_store()

        # -- setup data -- #
        self.setup_data()

        self.types_on = ['train']

        if self.val_on:
            self.types_on.append('val')

        if self.test_on:
            self.types_on.append('test')

        # -- split bars -- #
        self.split_bars(self.types_on)

        # -- loop types -- #
        for data_type in self.types_on:
            self.setup_stores(data_type)
            self.pre_build_datasets(data_type, 'model_bars', 'data_ds')

        return

    def create_store(self):
        return {'train': {'close_bars': None,
                          'model_bars': None,
                          'x_bars': None,
                          'y_bars': None,
                          'c_bars': None,
                          't_bars': None,
                          'data_ds': None,
                          'batch_data': None,
                          'input_shp': None,
                          'n_batches': None,
                          'raw': {'full_bars': None,
                                  'x_bars': None,
                                  'y_bars': None,
                                  'c_bars': None,
                                  't_bars': None,
                                  'data_ds': None},
                          'config': {
                              'batch_size': self.data.batch_size,
                              'n_days': self.data.train_days,
                              'n_batches': 0,
                              'n_steps': 0,
                              'n_samples': 0,
                              'n_calls': 0,
                              'masks': {'blocks': {'all_cols': [],
                                                   'x_cols': [],
                                                   'y_cols': [],
                                                   't_cols': []}}}},
                'test': {'close_bars': None,
                         'model_bars': None,
                         'data_ds': None,
                         'batch_data': None,
                         'input_shp': None,
                         'n_batches': None,
                         'raw': {'full_bars': None,
                                 'x_bars': None,
                                 'y_bars': None,
                                 'c_bars': None,
                                 't_bars': None,
                                 'data_ds': None},
                         'config': {
                             'batch_size': self.data.batch_size,
                             'n_days': self.data.test_days,
                             'n_batches': 0,
                             'n_steps': 0,
                             'n_samples': 0,
                             'n_calls': 0,
                             'masks': {'blocks': {'all_cols': [],
                                                  'x_cols': [],
                                                  'y_cols': [],
                                                  't_cols': []}}}},
                'val': {'close_bars': None,
                        'model_bars': None,
                        'data_ds': None,
                        'batch_data': None,
                        'input_shp': None,
                        'n_batches': None,
                        'raw': {'full_bars': None,
                                'x_bars': None,
                                'y_bars': None,
                                'c_bars': None,
                                't_bars': None,
                                'data_ds': None},
                        'config': {
                            'batch_size': self.data.batch_size,
                            'n_days': self.data.val_days,
                            'n_batches': 0,
                            'n_steps': 0,
                            'n_samples': 0,
                            'n_calls': 0,
                            'masks': {'blocks': {'all_cols': [],
                                                 'x_cols': [],
                                                 'y_cols': [],
                                                 't_cols': []}}}}}

    def setup_data(self):

        if self.data.seq_agg > 0:
            self.data.seq_depth = int(self.data.seq_len / self.data.seq_agg)

        if 'c_cols' in self.data.f_cols:
            self.data.input_shape = (
                self.data.seq_depth, self.data.n_x_cols + self.data.n_c_cols)

        if 'c_cols' not in self.data.f_cols:
            self.data.input_shape = (self.data.seq_depth, self.data.n_x_cols)

        self.data.targets_shape = (self.data.seq_depth, self.data.n_y_cols)
        self.data.tracking_shape = (self.data.seq_depth, self.data.n_t_cols)

    def split_bars(self, _types):

        train_days = self.data.store['train']['config']['n_days']
        test_days = self.data.store['test']['config']['n_days']
        val_days = self.data.store['val']['config']['n_days']

        load_days = train_days + test_days + val_days

        self.data.close_bars = self.data.full_bars[self.data.full_bars['is_close']
                                                   == True][self.data.close_cols].copy()

        max_days = self.data.full_bars['day_idx'].max()

        full_base = max_days - load_days - 2

        train_st_day = full_base
        train_ed_day = train_st_day + train_days

        test_st_day = train_ed_day + 1

        if 'test' in _types:
            test_ed_day = test_st_day + test_days

        if 'test' not in _types:
            test_ed_day = test_st_day + 1

        if 'val' in _types:
            val_st_day = test_ed_day + 1
            val_ed_day = val_st_day + val_days

        # --- reduce full bars by number of days --- #
        self.data.store['train']['full_bars'] = \
            self.data.full_bars[(self.data.full_bars['day_idx'] >= (train_st_day - self.data.seq_days)) &
                                (self.data.full_bars['day_idx'] < train_ed_day)].copy()

        if 'test' in _types:
            self.data.store['test']['full_bars'] = \
                self.data.full_bars[(self.data.full_bars['day_idx'] >= (test_st_day - self.data.seq_days)) &
                                    (self.data.full_bars['day_idx'] < test_ed_day)].copy()

        if 'val' in _types:
            self.data.store['val']['full_bars'] = \
                self.data.full_bars[(self.data.full_bars['day_idx'] >= (val_st_day - self.data.seq_days)) &
                                    (self.data.full_bars['day_idx'] < val_ed_day)].copy()

        # --- reduce close bars by number of days --- #

        self.data.store['train']['close_bars'] = \
            self.data.close_bars[(self.data.close_bars['day_idx'] >= train_st_day) &
                                 (self.data.close_bars['day_idx'] < train_ed_day)].copy()

        if 'test' in _types:
            self.data.store['test']['close_bars'] = \
                self.data.close_bars[(self.data.close_bars['day_idx'] >= test_st_day) &
                                     (self.data.close_bars['day_idx'] < test_ed_day)].copy()

        if 'val' in _types:
            self.data.store['val']['close_bars'] = \
                self.data.close_bars[(self.data.close_bars['day_idx'] >= val_st_day) &
                                     (self.data.close_bars['day_idx'] < val_ed_day)].copy()

    def setup_stores(self, data_type):

        self.data.store[data_type]['config']['n_samples'] = \
            self.data.samples_pd * \
            self.data.store[data_type]['config']['n_days']

        full_bars = self.data.store[data_type]['full_bars']

        # -- aggregate full bars -- #
        if self.data.seq_agg > 1:
            full_bars = self.agg_bars(full_bars)

        # -- normalise full bars -- #
        full_bars = self.normalize_bars(full_bars)

        if self.data.seq_days:
            self.data.store[data_type]['model_bars'] = self.seq_bars(
                full_bars, data_type)
        else:
            self.data.store[data_type]['model_bars'] = full_bars.to_numpy()

        self.data.store[data_type]['full_bars'] = full_bars

        if self.data.save_raw:
            self.data.store[data_type]['raw']['full_bars'] = full_bars.to_numpy()

        return

    def agg_bars(self, data_bars):

        n_bins = int(data_bars.shape[0] / self.data.seq_agg)

        data_bars = data_bars.groupby(pd.cut(data_bars.index, bins=n_bins)).agg(
            self.data.agg_data).dropna()

        return data_bars.reset_index(drop=True)

    def normalize_bars(self, data_bars):

        x_norm = self.data.preproc['normalize']['x_cols']
        c_norm = self.data.preproc['normalize']['c_cols']

        if x_norm is not None:

            # -- x groups off -- #
            if not self.network.x_groups_on:
                x_cols = self.data.nor_data['x']

                self.data.preproc_trans['train']['x_cols'] = getattr(
                    preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['x_cols'].fit(
                    data_bars[x_cols])
                data_bars[x_cols] = self.data.preproc_trans['train']['x_cols'].transform(
                    data_bars[x_cols])

            # -- x groups on -- #
            if self.network.x_groups_on:

                # -- pr cols -- #
                pr_cols = self.data.x_groups['pr_group']

                self.data.preproc_trans['train']['pr_cols'] = getattr(
                    preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['pr_cols'].fit(
                    data_bars[pr_cols])
                data_bars[pr_cols] = self.data.preproc_trans['train']['pr_cols'].transform(
                    data_bars[pr_cols])

                # -- vol cols -- #
                vol_cols = self.data.x_groups['vol_group']

                self.data.preproc_trans['train']['vol_cols'] = getattr(
                    preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['vol_cols'].fit(
                    data_bars[vol_cols])
                data_bars[vol_cols] = self.data.preproc_trans['train']['vol_cols'].transform(
                    data_bars[vol_cols])

                # -- atr cols -- #
                atr_cols = self.data.x_groups['atr_group']

                self.data.preproc_trans['train']['atr_cols'] = getattr(
                    preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['atr_cols'].fit(
                    data_bars[atr_cols])
                data_bars[atr_cols] = self.data.preproc_trans['train']['atr_cols'].transform(
                    data_bars[atr_cols])

                # -- roc cols -- #
                roc_cols = self.data.x_groups['roc_group']

                self.data.preproc_trans['train']['roc_cols'] = getattr(
                    preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['roc_cols'].fit(
                    data_bars[roc_cols])
                data_bars[roc_cols] = self.data.preproc_trans['train']['roc_cols'].transform(
                    data_bars[roc_cols])

                # -- zsc cols -- #
                zsc_cols = self.data.x_groups['zsc_group']

                self.data.preproc_trans['train']['zsc_cols'] = getattr(
                    preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['zsc_cols'].fit(
                    data_bars[zsc_cols])
                data_bars[zsc_cols] = self.data.preproc_trans['train']['zsc_cols'].transform(
                    data_bars[zsc_cols])

        # --- c cols --- #
        if c_norm is not None and self.data.n_c_cols > 0:
            c_cols = self.data.nor_data['c']

            self.data.preproc_trans['train']['c_cols'] = getattr(
                preprocessing, c_norm['cls'])(**c_norm['params'])
            self.data.preproc_trans['train']['c_cols'].fit(data_bars[c_cols])
            data_bars[c_cols] = self.data.preproc_trans['train']['c_cols'].transform(
                data_bars[c_cols])

        return data_bars

    def seq_bars(self, full_bars, data_type):

        model_bars = []

        close_bars = self.data.store[data_type]['close_bars']

        n_bars = close_bars.shape[0]

        # --- append seq bars to close bars to generate model bars --- #
        for _idx in range(n_bars):
            _bar = close_bars.iloc[_idx]

            _st_idx = _bar['bar_idx'] - self.data.seq_len
            _ed_idx = _bar['bar_idx']

            seq_bars = full_bars[(full_bars['bar_idx'] > _st_idx) &
                                 (full_bars['bar_idx'] <= _ed_idx)]

            model_bars.append(seq_bars.to_numpy())

        # -- concat model bars -- #
        model_bars = np.concatenate(model_bars, axis=0)

        # --- reshape seq bars ---  #
        n_samples = self.data.store[data_type]['config']['n_samples']

        seq_depth = int(model_bars.shape[0] / n_samples)

        _new_shp = (n_samples, seq_depth, model_bars.shape[-1])

        return np.reshape(model_bars, _new_shp)

    def setup_outputs_ds(self, data_type):

        x_bars = self.data.store[data_type]['model_bars'][self.data.slice_configs['x_slice']]

        return x_bars
    # tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_bars,
    #                                                                    dtype=self.data.dtype))

    def get_dataset(self,
                    data,
                    data_type,
                    batch_size):

        store = data.store[data_type]

        if data.framework == 'tf':

            x_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(store['x_bars'], dtype=data.dtype))
            y_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(store['y_bars'], dtype=data.dtype))
            t_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(store['t_bars'], dtype=data.dtype))

            if data.outputs_on:
                o_ds = tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(store['model_bars'][data.slice_configs['x_slice']], dtype=data.dtype))
                return tf.data.Dataset.zip((x_ds, y_ds, t_ds, o_ds)).batch(batch_size)
            else:
                return tf.data.Dataset.zip((x_ds, y_ds, t_ds)).batch(batch_size)

        elif data.framework == 'torch':

            x_bars = torch.tensor(store['x_bars'], dtype=torch.float32)
            y_bars = torch.tensor(store['y_bars'], dtype=torch.float32)
            t_bars = torch.tensor(store['t_bars'], dtype=torch.float32)

            if data.outputs_on:
                o_bars = torch.tensor(store['model_bars'][data.slice_configs['x_slice']], dtype=data.dtype)
                ds = TensorDataset(x_bars, y_bars, t_bars, o_bars)
            else:
                ds = TensorDataset(x_bars, y_bars, t_bars)

            return DataLoader(ds, batch_size=batch_size, shuffle=False)

        else:
            raise ValueError

    def pre_build_datasets(self, data_type, src, dest):

        batch_size = self.data.store[data_type]['config']['batch_size']

        n_batches = int(self.data.store[data_type][src].shape[0] / batch_size)

        self.data.store[data_type]['n_batches'] = n_batches

        # -- splts -- #

        self.data.store[data_type]['x_bars'] = self.data.store[data_type][src][self.data.slice_configs['x_slice']]
        self.data.store[data_type]['c_bars'] = self.data.store[data_type][src][self.data.slice_configs['c_slice']]
        self.data.store[data_type]['y_bars'] = self.data.store[data_type][src][self.data.slice_configs['y_slice']]
        self.data.store[data_type]['t_bars'] = self.data.store[data_type][src][self.data.slice_configs['t_slice']]

        if 'c_cols' in self.data.f_cols:
            self.data.store[data_type]['x_bars'] = \
                np.concatenate([self.data.store[data_type]['x_bars'],
                                self.data.store[data_type]['c_bars']], axis=-1)

        out_ds = self.setup_outputs_ds(data_type)

        self.data.store[data_type][dest] = self.get_dataset(self.data, data_type, batch_size)

        return
