import numpy as np
import tensorflow as tf

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

from photon.gauge import Gauge


class Chains():

    def __init__(self, branch, chain_idx):

        self.run_device_types = 'model_gpus'

        self.is_built = False

        self.branch = branch
        self.chain_idx = chain_idx

        self.network = self.branch.network
        self.network.chains.append(self)

        self.photon = self.branch.photon

        self.trees = []

        for tree in self.branch.trees:

            tree.chains.append(self)
            self.trees.append(tree)

        self.name = self.branch.name + '_chain_' + str(chain_idx)

        self.models = []

        self.datasets = []

        self.idx_gen = []

        self.input_data = None

        self.model_gpus = []

        self.model_subs = []

        self.logs = self.Logs()

    def build_chain(self):

        self.model_config = self.branch.configs.by_chain_idx(
            'model', self.chain_idx)
        self.data_config = self.branch.configs.by_chain_idx(
            'data', self.chain_idx)
        self.build_config = self.branch.configs.by_chain_idx(
            'build', self.chain_idx)

        self.opt_config = self.branch.configs.by_chain_idx(
            'opt', self.chain_idx)
        self.loss_config = self.branch.configs.by_chain_idx(
            'loss', self.chain_idx)
        self.metrics_config = self.branch.configs.by_chain_idx(
            'metrics', self.chain_idx)
        self.save_config = self.branch.configs.by_chain_idx(
            'save', self.chain_idx)

        self.n_models = self.model_config['n_models']
        self.n_outputs = self.model_config['n_models']

        # -- setup data slices -- #
        self.setup_slices()

        # -- loop trees -- #
        for tree_idx, tree in enumerate(self.trees):

            # -- load data -- #
            tree.load_data()

            # --- setup datasets --- #
            train_ds = tree.data.store['train']['data_ds']
            val_ds = None
            test_ds = None

            if tree.data.store['val']['data_ds']:
                val_ds = tree.data.store['val']['data_ds']

            if tree.data.store['test']['data_ds']:
                test_ds = tree.data.store['test']['data_ds']

            self.datasets.insert(tree_idx, {'train': train_ds,
                                            'val': val_ds,
                                            'test': test_ds})

        # -- build input data placeholder -- #
        if self.input_data is None:

            self.input_data = tf.keras.Input(shape=self.trees[0].data.input_shape,
                                             batch_size=self.trees[0].data.batch_size,
                                             dtype=self.network.float_x,
                                             name=self.name + '_input_data')

            self.targets_data = tf.keras.Input(shape=self.trees[0].data.targets_shape,
                                               batch_size=self.trees[0].data.batch_size,
                                               dtype=self.network.float_x,
                                               name=self.name + '_targets_data')

            self.tracking_data = tf.keras.Input(shape=self.trees[0].data.tracking_shape,
                                                batch_size=self.trees[0].data.batch_size,
                                                dtype=self.network.float_x,
                                                name=self.name + '_tracking_data')

        # --- setup gauge/models --- #
        for model_idx in range(self.n_models):

            # --- init gauge --- #
            gauge = Gauge(chain=self, model_idx=model_idx)

            # -- insert into chain models -- #
            self.models.insert(model_idx, gauge)

        self.is_built = True

        return

    def setup_slices(self):

        targets_config = self.data_config['targets']
        split_on = targets_config['split_on']

        self.data_config['targets']['true_slice'] = np.s_[..., :split_on]
        self.data_config['targets']['tracking_slice'] = np.s_[..., split_on:]

    @dataclass
    class Logs:

        batch_data: List = field(default_factory=lambda: {
                                 'main': [[]], 'val': [[]]})
