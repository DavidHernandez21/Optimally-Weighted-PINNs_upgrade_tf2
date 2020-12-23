from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from Scripts_tf2.main_laplace_2d import main as main_laplace_2d
from Scripts_tf2.main_laplace_nd import main as main_laplace_nd

plt.rcParams.update({'font.size': 16})
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

main_laplace_2d()
# main_laplace_nd()
