module purge
module load Stages/2025
module load GCC OpenMPI

module load jupyter-server

# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn OpenCV

# ML Frameworks
module load PyTorch scikit-learn torchvision PyTorch-Lightning
module load tensorboard
