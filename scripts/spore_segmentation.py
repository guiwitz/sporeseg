import argparse
from spores.sporeClass import Spore

parser=argparse.ArgumentParser()
parser.add_argument('exp_folder', type = str, help='Experiment folder')
parser.add_argument('result_folder', type = str, help='Result storage folder')
parser.add_argument('--convexity', type = float, help='Convexity threshold')
parser.add_argument('--show_output', action='store_true', help='show plots')
args=parser.parse_args()

spores = Spore()

exp_folder = args.exp_folder
result_folder = args.result_folder

spores.convexity = args.convexity
spores.show_output = args.show_output

spores.analyse_spore_folder(exp_folder, result_folder)

