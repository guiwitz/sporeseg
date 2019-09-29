import sys
import os

from sporeClass import Spore

spores = Spore(show_output=False)

main_folder = sys.argv[1]
result_folder = sys.argv[2]

folders = os.listdir(main_folder)
folders = [x for x in folders if x[0]!='.']

for f in folders:
    current_folder = os.path.normpath(main_folder)+'/'+f
    spores.analyse_spore_folder(current_folder, result_folder)
    spores.split_categories(spores.path_to_analysis(current_folder, result_folder))
    spores.plot_image_categories(current_folder, result_folder)



