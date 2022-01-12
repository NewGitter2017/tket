# Copyright 2019-2021 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for splitting circuits to a variable number of subcircuits and 
converting each to measurement based quantum computing."""

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import patches, lines, path
#import matplotlib
#from pyzx.gflow import gflow as gflow
#import pyzx as zx
#import pytket
#import pytket.pyzx
#import random as rng
#import math
#from fractions import Fraction
#from statistics import mean as mean
#try:
#    import cPickle as pickle
#except ModuleNotFoundError:
#    import pickle

#from .circuit import Qubit, Circuit, OpType, fresh_symbol, PauliExpBox
#from .utils import gen_term_sequence_circuit
#from .utils import QubitPauliOperator
#rom .transform import Transform, PauliSynthStrat, CXConfigType
#from .pauli import Pauli
    
#from scipy.optimize import curve_fit