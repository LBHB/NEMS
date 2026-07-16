"""Collects commonly-used core classes & functions for easier import.

Ex: `from nems import Model`

"""

import nems.registry
from nems.models.base import Model, Model_List
from nems.models.LN import LN_STRF, LN_pop
from nems.models.CNN import CNN_pop
from nems.models.decoder import binary

from nems.tools.demo_data.file_management import download_demo, load_demo
from nems.tools import log