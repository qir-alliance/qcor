import sys, os
from pathlib import Path
sys.path.insert(1, str(Path.home()) + "/.xacc")

from qcor import *
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate Heisenberg model (ArQTiC format) input
problemModel = qsim.ModelBuilder.createModel(ModelType.Heisenberg, {'Jz': 0.01183898,
                                                       'h_ext': 0.01183898,
                                                       'freq': 0.0048,
                                                       'ext_dir': 'X',
                                                       'num_spins': 3})
# Run TD workflow:
workflow = qsim.getWorkflow(
      'td-evolution', {'dt': 3.0, 'steps': 20})
result = workflow.execute(problemModel)

# Plot the result:
t = np.linspace(0, 60, 21)
y = result["exp-vals"]
axes = plt.axes()
axes.plot(t, y)
axes.set_xlim([0,60])
axes.set_ylim([0,1])
# Save the plot to a file:
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.savefig("avg_magnetization_plot.pdf")  