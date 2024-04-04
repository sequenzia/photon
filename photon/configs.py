from __future__ import annotations

import os

from typing import List, Dict, Optional, Union, Literal, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

import torch


@dataclass
class Config():

    run_dir: str = field(default="")

    framework: Literal["torch", "tf"] = field(default="torch")

    precision: int = field(default=32)

    def __post_init__(self):

        if not self.run_dir:
            self.run_dir = os.path.expanduser('~') + '/photon_temp'

        if self.precision == 16:

            self.float_x = 'float16'
            self.dtype = np.float16
            self.framework_dtype = torch.float16 if self.framework == 'torch' else tf.float16

        elif self.precision == 32:

            self.float_x = 'float32'
            self.dtype = np.float32
            self.framework_dtype = torch.float32 if self.framework == 'torch' else tf.float32
        
        else:

            self.float_x = 'float64'
            self.dtype = np.float64
            self.framework_dtype = torch.float64 if self.framework == 'torch' else tf.float64
