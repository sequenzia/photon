from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from photon.chains import Chains


class Branches():

    def __init__(self,
                 trees,
                 n_epochs,
                 n_chains,
                 model_config,
                 data_config,
                 build_config,
                 opt_config,
                 loss_config,
                 metrics_config,
                 run_config,
                 save_config,
                 name=None,
                 **kwargs):

        self.is_built = False

        self.trees = trees

        self.network = self.trees[0].network
        self.photon = self.network.photon

        if name is None:
            self.name = 'Photon Branch'
        else:
            self.name = name

        self.n_epochs = n_epochs

        self.n_chains = n_chains
        self.chains = []

        self.msgs_on = False

        self.run = None

        # -- turn on branch msgs -- #
        for rc in run_config:
            if rc['msgs_on']:
                self.msgs_on = True

        # -- configs -- #
        self.configs = self.Configs(model_config=model_config,
                                    data_config=data_config,
                                    build_config=build_config,
                                    opt_config=opt_config,
                                    loss_config=loss_config,
                                    metrics_config=metrics_config,
                                    run_config=run_config,
                                    save_config=save_config)

        # --- add branch to network --- #
        self.branch_idx = self.network.add_branch(self)

        # --- loop chains to init/build --- #
        for chain_idx in range(self.n_chains):

            # -- init chains -- #
            chain = Chains(self, chain_idx)

            if not chain.is_built:
                chain.build_chain()

            self.chains.insert(chain_idx, chain)

    @dataclass()
    class Configs:

        model_config: List
        data_config: List
        build_config: List
        opt_config: List
        loss_config: List
        metrics_config: List
        run_config: List
        save_config: List

        def by_chain_idx(self, type: str, chain_idx: int) -> Dict:

            obj = getattr(self, type+'_config')

            if type == 'metrics':
                obj = [obj]

            if len(obj) <= chain_idx:
                obj_out = obj[-1]
            else:
                obj_out = obj[chain_idx]

            if len(obj) <= chain_idx:
                obj_out = obj[-1]
            else:
                obj_out = obj[chain_idx]

            return obj_out
