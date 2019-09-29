import sys
from sporeClass import Spore

spores = Spore(show_output=False)

exp_folder = sys.argv[1]
result_folder = sys.argv[2]

spores.analyse_spore_folder(exp_folder, result_folder)

