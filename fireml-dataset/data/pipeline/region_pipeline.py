import os
import numpy as np

from dagma import create_node

import fireml.data.regions as reg

def load_region(data_dir, region_name, region_padding):
    return reg.load_region(data_dir, region_name, region_padding)


def get_region():
    return load_region("data_dir", "region", "region_padding")
