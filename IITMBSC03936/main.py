### Imports ###
# add imports - classes and defs
from predictor import *


"""
sys.argv[1] is the input test file name given as command line arguments

"""
runs = predictRuns('inputFile_1.csv')
print("Predicted Runs: ", runs)