import os
from pathlib import Path
import shutil

import skimage.io
import skimage.morphology
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sporeseg.spores import Spore

# create cropped image for fast tests
if not Path('sporeseg/tests/Sporefolder/small').is_dir():
    os.mkdir(Path('sporeseg/tests/Sporefolder/small'))

# load small test images
image = skimage.io.imread(Path('sporeseg/tests/Sporefolder/SporesA/spore2.jpg'))[500:850,550:900,:]
skimage.io.imsave(Path('sporeseg/tests/Sporefolder/small/small_spore2.jpg'), image, check_contrast=False)
image2 = skimage.io.imread(Path('sporeseg/tests/Sporefolder/SporesA/spore1.jpg'))[1400:1750,800:1150,:]
skimage.io.imsave(Path('sporeseg/tests/Sporefolder/small/small_spore1.jpg'), image, check_contrast=False)

# define data locations
exp_folder = 'sporeseg/tests/Sporefolder/SporesA/'
exp_file = 'sporeseg/tests/Sporefolder/SporesA/spore1.jpg'
result_folder = 'sporeseg/tests/Results'
exp_folder_small = 'sporeseg/tests/Sporefolder/small'
exp_file_small = 'sporeseg/tests/Sporefolder/small/small_spore2.jpg'

# instantiate class
spores = Spore(show_output=False)

def test_path_to_analysis_folder():
    """test that correct result folder is created for folder analysis"""
    
    output = spores.path_to_analysis(exp_folder, result_folder)
    assert output == Path('sporeseg/tests/Results/SporesA'), f'expected path is "sporeseg/tests/Results/Spores" but got {output}'

    output = spores.path_to_analysis(Path(exp_folder), Path(result_folder))
    assert output == Path('sporeseg/tests/Results/SporesA'), f'expected path is "sporeseg/tests/Results/Spores" but got {output}'

def test_path_to_analysis_file():
    """test that correct result folder is created for file analysis"""
    
    output = spores.path_to_analysis(exp_file, result_folder)
    assert output == Path('sporeseg/tests/Results/SporesA'), f'expected path is "sporeseg/tests/Results/SporesA" but got {output}'
    
    output = spores.path_to_analysis(Path(exp_file), Path(result_folder))
    assert output == Path('sporeseg/tests/Results/SporesA'), f'expected path is "sporeseg/tests/Results/SporesA" but got {output}'

def test_segmentation():
    """test that correct number of objects are detected"""
    
    image_load, newmask = spores.segmentation(exp_file_small)
    num_obj = np.max(skimage.morphology.label(newmask))

    assert isinstance(image_load, np.ndarray), "second output is not an ndarray"
    assert isinstance(newmask, np.ndarray), "second output is not an ndarray"
    assert newmask.max() == 1.0, "mask objects not equal to 1.0"
    assert image_load.shape[0] == 350
    assert image_load.shape[1] == 350
    assert num_obj == 7, f"wrong number of objects detected, found {num_obj}, expects 7"
    
def test_plot_segmentation():
    """test the figure is generated"""
    
    regions_prop, raw_im, image_mask = spores.find_spores(exp_file_small)
    fig = spores.plot_segmentation(raw_im, image_mask)
    assert isinstance(fig, matplotlib.figure.Figure)
    
    
def test_analyse_single_image():
    """test that all outputs are generated"""
    
    # test with file
    if Path(result_folder).is_dir():
        shutil.rmtree(result_folder)
    spores.analyse_single_image(exp_file_small, result_folder)
    assert Path('sporeseg/tests/Results/small/small_spore2.csv').is_file(), 'no csv export'
    assert Path('sporeseg/tests/Results/small/small_spore2.pkl').is_file(), 'no pkl export'
    assert Path('sporeseg/tests/Results/small/small_spore2_seg.png').is_file(), 'no png export'

    # test with file name
    spores.analyse_single_image(exp_file_small, result_folder, save_name='mytest')
    assert Path('sporeseg/tests/Results/mytest/mytest.csv').is_file(), 'no csv export'
    assert Path('sporeseg/tests/Results/mytest/mytest.pkl').is_file(), 'no pkl export'
    assert Path('sporeseg/tests/Results/mytest/mytest_seg.png').is_file(), 'no png export'
    
    # test with image
    spores.analyse_single_image(image[:,:,0], result_folder)
    assert Path('sporeseg/tests/Results/image/image.csv').is_file(), 'no csv export'
    assert Path('sporeseg/tests/Results/image/image.pkl').is_file(), 'no pkl export'
    assert Path('sporeseg/tests/Results/image/image_seg.png').is_file(), 'no png export'
    
def test_find_spores():
    """test that detected objects have ~ correct properties and that image outputs have correct type"""
    
    regions_prop, raw_im, image_mask = spores.find_spores(exp_file_small)

    assert len(regions_prop) == 7, "found {len(regions_prop)} spores, should be 7"
    assert regions_prop.iloc[0].filename == "sporeseg/tests/Sporefolder/small/small_spore2.jpg", "bad file name in table"
    assert regions_prop.area.max() < 800, "unexpectedly found a spore > 800 pixels"
    assert regions_prop.area.max() > 550, "unexpectedly found a spore < 550 pixels"
    assert isinstance(raw_im, np.ndarray), "second output is not an ndarray"
    assert isinstance(image_mask, np.ndarray), "second output is not an ndarray"
    assert image_mask.max() == 1.0, "mask objects not equal to 1.0"
    
def test_analyze_spore_folder():
    """test that correct number of files are analyzed in a folder"""

    spores.analyse_spore_folder(exp_folder_small, result_folder)

    assert len(list(Path('sporeseg/tests/Results/small/').glob('*.csv'))) == 2, "not all csv files have been analyzed"
    assert len(list(Path('sporeseg/tests/Results/small/').glob('*.png'))) == 2, "not all png files have been analyzed"
    assert len(list(Path('sporeseg/tests/Results/small/').glob('*.pkl'))) == 2, "not all pkl files have been analyzed"

def test_load_experiment():
    """test that re-loading of data loads all data"""

    output_df = spores.load_experiment(Path('sporeseg/tests/Results/small/'))
    assert len(output_df) == 14, f"Incomplete data. Loaded {len(output_df)} lines but expects 14"
    

def test_split_categories():
    """test that round spore detection with threshold generates correct number of positives"""

    spores.threshold = 0.7
    spores.split_categories(Path('sporeseg/tests/Results/small/'))

    assert Path('sporeseg/tests/Results/small/small_summary.csv').is_file(), 'no summary csv export'
    assert Path('sporeseg/tests/Results/small/small_summary.png').is_file(), 'no summary png export'
    
    summary = pd.read_csv(Path('sporeseg/tests/Results/small/small_summary.csv'))
    assert summary['round'].sum() == 4, f"Wrong number of round cells. Found {summary['round'].sum()}, expects 4"
