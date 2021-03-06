import os
import argparse
from sporeseg.spores import Spore

parser = argparse.ArgumentParser()
parser.add_argument('main_folder', type=str, help='General folder containing multiple experiments')
parser.add_argument('result_folder', type=str, help='Result storage folder')
parser.add_argument('--max_area', type=int, help='Max area to consider')
parser.add_argument('--min_area', type=int, help='Min area to consider')
parser.add_argument('--threshold', type=float, help='Classification threshold')
parser.add_argument('--convexity', type=float, help='Convexity threshold')
parser.add_argument('--bin_width', type=float, help='Histogram bin width')
parser.add_argument('--show_output', action='store_true', help='show plots')
parser.add_argument('--show_title', action='store_true', help='show title')
parser.add_argument('--show_legend', action='store_true', help='show legend')
args = parser.parse_args()

spores = Spore()

main_folder = args.main_folder
result_folder = args.result_folder

if args.min_area is not None:
    spores.min_area = args.min_area
if args.max_area is not None:
    spores.max_area = args.max_area
if args.bin_width is not None:
    spores.bin_width = args.bin_width
if args.convexity is not None:
    spores.convexity = args.convexity

spores.threshold = args.threshold
spores.show_output = args.show_output
spores.show_title = args.show_title
spores.show_legend = args.show_legend

folders = os.listdir(main_folder)
folders = [x for x in folders if x[0] != '.']

for f in folders:
    current_folder = os.path.normpath(main_folder)+'/'+f
    spores.analyse_spore_folder(current_folder, result_folder)
    spores.split_categories(spores.path_to_analysis(current_folder, result_folder))
    spores.plot_image_categories(current_folder, result_folder)
