"""Collects commonly-used core classes & functions for easier import.

Ex: `from nems import Model`

"""

from nems.tools.cuda_libs import preload_cuda_libs  # [AGENT EDIT | agent: claude-opus-5-5 | user: svd | reason: GPU without conda activate (TF 2.21 cusolver RUNPATH gap) | date: 2026-11-27]
preload_cuda_libs()                                  # [AGENT EDIT | agent: claude-opus-5-5 | user: svd | reason: GPU without conda activate (TF 2.21 cusolver RUNPATH gap) | date: 2026-11-27]

import nems.registry
from nems.models.base import Model, Model_List
from nems.models.LN import LN_STRF, LN_pop
from nems.models.CNN import CNN_pop
from nems.models.decoder import binary

from nems.tools.demo_data.file_management import download_demo, load_demo
from nems.tools import log