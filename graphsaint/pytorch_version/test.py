import torch
import torch
import scipy as sp
import numpy as np
import argparse
from graphsaint.kgraphsaint import loader
from graphsaint.graph_samplers import *
import time

print(torch.cuda.is_available())
