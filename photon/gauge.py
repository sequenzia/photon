import os
import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

from photon.theta import Theta


class Gauge():

    def __init__(self, chain, model_idx):

        self.is_built = False
        self.is_compiled = False

        self.is_model_built = False

        self.chain = chain
        self.model_idx = model_idx

        self.chain_idx = self.chain.chain_idx
        self.branch = self.chain.branch
        self.network = self.branch.network
        self.dtype = self.network.dtype

        self.name = chain.name + '_model_' + str(model_idx)

        self.input_shape = ()

        self.layers = {}
        self.parent_layers = {}
        self.child_layers = {}

        self.src = None

        self.model_args = None
        self.model_inputs = None

        self.opt_fn = None
        self.loss_fn = None
        self.metrics_fn = None

        self.conn_chain_idx = None
        self.conn_n_models = None

        self.chkp_manager = None
        self.chkp_dir = None

        self.run_chain = None
        self.run_model = None
        self.run_data = None
        self.is_live = False

        self.strat_on = False
        self.dist_on = False

        self.runs = []

        self.batch_data = []
        self.val_batch_data = []

        self.datasets = {'train': None,
                         'val': None,
                         'test': None}

        self.log_theta = False
        self.logs = self.Logs()

        self.setup_strats()

    def setup_strats(self):

        self.strat_type = self.chain.build_config['strat_type']
        self.dist_type = self.chain.build_config['dist_type']

        # self.run_device = '/GPU:0'
        self.strat = None

        if self.chain.photon.n_gpus > 1:

            # if self.strat_type is None:
            # self.run_device = self.chain.model_gpus[self.model_idx]['v_run_device']

            if self.strat_type is not None:

                self.strat_on = True

                if self.strat_type == 'Mirrored':
                    self.strat = tf.distribute.MirroredStrategy(
                        self.chain.photon.gpus)
                    self.dist_on = True

        # if self.strat_type == 'One':
        #
        #     self.strat_on = True
        #     self.strat = tf.distribute.OneDeviceStrategy(self.run_device)

    def build_gauge(self, run, chain, model):

        self.tree = self.chain.trees[0]

        self.model_args = self.chain.model_config['args']

        # -- init model -- #
        self.src = self.chain.model_config['model'](**{'gauge': self})

        # -- opt_fn -- #
        self.opt_fn = self.chain.opt_config['fn']

        # -- setup loss fn -- #
        self.setup_loss_fn()

        self.is_built = True

        self.runs.insert(run.run_idx, run)

        self.run_chain = chain
        self.run_model = model

        return

    def setup_loss_fn(self):

        self.loss_fn = self.chain.loss_config['fn'](
            **self.chain.loss_config['args'])

    def compile_gauge(self, tree_idx):

        self.opt_fn = self.opt_fn(gauge=self,
                                  tree=self.chain.trees[tree_idx],
                                  config=self.chain.opt_config['args'],
                                  n_epochs=self.branch.n_epochs)

        self.metrics_fns = []

        for config in self.chain.metrics_config:
            self.metrics_fns.append(config['fn'](config['args']))

        # -- compile model -- #
        self.src.compile(optimizer=self.opt_fn)

        self.log_theta = self.model_args['log_config']['log_theta']

        self.theta = Theta(self)

        self.is_compiled = True

    def setup_cp(self, step_idx, load_cp):

        if not self.is_compiled:
            self.compile_gauge()

        self.chkp_nm = self.chain.name + '_model_' + str(self.model_idx)

        self.chkp_dir = self.get_chkp_dir()

        self.chkp = tf.train.Checkpoint(model=self.src,
                                        step_idx=step_idx)

        self.chkp_manager = tf.train.CheckpointManager(checkpoint=self.chkp,
                                                       directory=self.chkp_dir,
                                                       max_to_keep=5,
                                                       step_counter=step_idx,
                                                       checkpoint_name=self.chkp_nm,
                                                       checkpoint_interval=1)

        if self.chkp_manager.latest_checkpoint and load_cp:

            chkp_status = None
            chkp_status = self.chkp.restore(
                self.chkp_manager.latest_checkpoint)
            chkp_status.assert_existing_objects_matched()

            print(f'model restored from {self.chkp_manager.latest_checkpoint}')

    def get_chkp_dir(self):

        branch_nm = 'branch_' + str(self.branch.branch_idx)

        chkp_dir = self.network.photon.store['chkps'] + '/' + self.network.photon.photon_nm.lower(
        ) + '/' + branch_nm + '/' + self.chain.name.lower() + '/model_' + str(self.model_idx)

        if self.network.photon_load_id == 0 and os.path.exists(chkp_dir):
            os.remove(chkp_dir)

        return chkp_dir

    def setup_run(self, run, chain, model):

        self.runs.insert(run.run_idx, run)

        self.run_chain = chain
        self.run_model = model

    def pre_build_model(self):

        if not self.is_compiled:
            self.compile_gauge()

        self.src.pre_build(input_data=self.chain.input_data,
                           targets_data=self.chain.targets_data,
                           tracking_data=self.chain.tracking_data)

    @dataclass
    class Logs:

        calls: List = field(default_factory=lambda: {
                            'main': [[]], 'val': [[]]})
        layers: List = field(default_factory=lambda: {
                             'main': [[]], 'val': [[]]})
        run_data: List = field(default_factory=lambda: {
                               'main': [[]], 'val': [[]]})
        theta: List = field(default_factory=lambda: [[]])
