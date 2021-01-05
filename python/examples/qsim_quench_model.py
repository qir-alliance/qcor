import sys, os
from pathlib import Path
sys.path.insert(1, str(Path.home()) + "/.xacc")

from qcor import *
import numpy as np
import matplotlib.pyplot as plt

# Heisenberg quench model input
problemModel = qsim.ModelFactory.createModel(ModelType.Heisenberg, {'Jx': 1.0,
                                                                    'Jy': 1.0,
                                                                    'Jz': 0.2,
                                                                    'h_ext': 0.0,
                                                                    'num_spins': 7,
                                                                    'initial_spins': [0, 1, 0, 1, 0, 1, 0],
                                                                    'observable': 'staggered_magnetization'})
print(problemModel)

# Run TD workflow:
workflow = qsim.getWorkflow(
      'td-evolution', {'dt': 0.05, 'steps': 100})
result = workflow.execute(problemModel)

# Plot the result:
t = np.linspace(0, 5, 101)
y = result["exp-vals"]
result_save = np.asarray(y)
result_save.tofile('run_g_0.2.csv',sep=',',format='%10.5f')
axes = plt.axes()
axes.plot(t, y)
axes.set_xlim([0,5])
# axes.set_ylim([0,1])
# Save the plot to a file:
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.savefig("staggered_magnetization_plot.pdf")  