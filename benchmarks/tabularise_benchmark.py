import timeit

SETUP = """
import numpy as np
import pandas as pd
{packages}
from sktime.datasets import load_gunpoint

# Create different sizes of data
X_base = load_gunpoint(return_X_y=False)
X_small = {X}
X_medium = pd.concat([X_small for _ in range(20)])
X_large = pd.concat([X_small for _ in range(200)])

# Choose the size used for this experiment
X = X_{size}.copy()

# Extract y into a separate vector
y = X['class_val']
X.drop('class_val', axis=1, inplace=True)
"""

repeats = 5
runs = 0


# **********************************************************************************************************************
# Time the tabularisation

PACKAGES = """
from sktime.transformers.compose import Tabulariser
"""

TEST = """
tab = Tabulariser()
X_tab = tab.fit_transform(X)
"""

timeit.repeat(setup=SETUP.format(packages=PACKAGES, X="X_base", size="medium"),
              stmt=TEST,
              repeat=repeats,
              number=runs)