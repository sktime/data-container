import timeit

SETUP_CODE = """
import pandas as pd
import numpy as np
from sktime.datasets import load_gunpoint
from sktime.transformers.compose import Tabulariser

# Load the original dataset and create different sizes
X_small, _ = load_gunpoint(return_X_y=False)
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])
"""

SEPARATE_Y = """
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)
"""

CHOOSE_X = """
X = X_medium.copy()
"""

repeats = 5
runs = 0


# **********************************************************************************************************************
# Time the tabularisation

TEST_CODE = """
tab = Tabulariser()
X_tab = tab.fit_transform(X)
"""

timeit.repeat(setup=SETUP_CODE + CHOOSE_X + SEPARATE_Y, stmt=TEST_CODE, repeat=repeats, number=runs)