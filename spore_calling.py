import sys
from sporeClass import Spore

spores = Spore(show_output=False)

exp_folder = sys.argv[1]
result_folder = sys.argv[2]

spores.split_categories(spores.path_to_analysis(exp_folder, result_folder))
spores.plot_image_categories(exp_folder, result_folder)

