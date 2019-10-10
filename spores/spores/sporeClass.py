import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.morphology
import skimage.segmentation
from sklearn import mixture
#from scipy.optimize import curve_fit
import matplotlib
cmap = matplotlib.colors.ListedColormap (np.random.rand(256,3))
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }


class Spore:
    
    def __init__(self, show_output = False, min_area = 250, max_area = 1000, 
                 show_title = True, show_legend = True, threshold = None):

        """Standard __init__ method.

        Parameters
        ----------
        show_output : bool
            show segmentation images during analysis
        show_title : bool
            add a title to histogram
        show_legend : bool
            add a legend to histogram
        min_area : int
            minimal area of spores considered in analysis
        max_area : int
            maximal area of spores considered in analysis
        threshold = float
            fixed threshold to use for splitting
        """

        self.show_output = show_output
        self.show_title = show_title
        self.show_legend = show_legend
        self.min_area = min_area
        self.max_area = max_area
        self.threshold = threshold

    #create anad get the experiment specific folder stored in the result_folder
    def path_to_analysis(self, data_path, result_folder):

        if os.path.isfile(data_path):
            result_folder_exp = result_folder+'/'+os.path.basename(os.path.dirname(data_path))
        else:
            result_folder_exp = os.path.normpath(result_folder)+'/'+os.path.basename(os.path.normpath(data_path))

        if not os.path.isdir(result_folder_exp):
            os.makedirs(result_folder_exp, exist_ok=True)

        return result_folder_exp

    #segment single image stored in file. Save result in result_folder
    def analyse_single_image(self, image_path, result_folder):
        regions, image, image_seg = self.find_spores(image_path)
        fig = self.plot_segmentation(image, image_seg)

        result_folder_exp = self.path_to_analysis(image_path, result_folder)

        regions.to_pickle(result_folder_exp+'/'+os.path.basename(image_path).split('.')[0]+'.pkl')
        regions[['area','ecc']].to_csv(result_folder_exp+'/'+os.path.basename(image_path).split('.')[0]+'.csv',index = False)
        fig.savefig(result_folder_exp+'/'+os.path.basename(image_path).split('.')[0]+'_seg.png', dpi = 300)
        plt.close(fig)
        #return regions, image, image_seg

    #segment spores in a given image and measure their properties
    def find_spores(self, image_path):

        #get a binary segmentation
        raw_im, image = self.segmentation(image_path)
        
        #measure region properties and keep area, eccentricity and coords
        regions = skimage.measure.regionprops(skimage.morphology.label(image),coordinates='rc')

        regions_prop = pd.DataFrame({'area':[x.area for x in regions],'ecc':[x.eccentricity for x in regions],
                                    'coords': [x.coords for x in regions]})
        regions_prop['filename'] = image_path

        return regions_prop, raw_im, image

    #create binary mask of spores
    def segmentation(self, image_path):
        #import image
        raw_image = skimage.io.imread(image_path)
        #do a median filtering to remove outliers
        raw_image = skimage.filters.median(raw_image[:,:,0],selem=skimage.morphology.disk(10))
        #do a large scale background subtraction by using a very large gaussian
        flatten = raw_image-skimage.filters.gaussian(raw_image,sigma=100,preserve_range=True)
        
        #calculate a binary mask using automatic Li thresholding
        #mask = flatten<skimage.filters.threshold_li(flatten)
        
        #calcualte a binary mask by using the background distribution (asssumes low density of spores)
        mask = flatten < np.mean(flatten)-2*np.std(flatten)
        
        #remove objects touching the border
        mask = skimage.segmentation.clear_border(mask)

        return raw_image, mask

    #plot segmentation result for an image
    def plot_segmentation(self, image, image_seg, saving = True):
        image_seg = image_seg.astype(float)
        image_seg[image_seg == 0] = np.nan
        fig,ax = plt.subplots(figsize=(10,10))
        plt.imshow(image,cmap = 'gray')
        plt.imshow(image_seg,cmap = 'Reds',alpha = 0.7,vmin=0,vmax = 1.5)
        ax.set_axis_off()
        plt.tight_layout()
        if self.show_output:
            plt.show()
        return fig


    #segment all images stored in folder. Save result in result_folder
    def analyse_spore_folder(self, exp_folder, result_folder):
        filenames = glob.glob(os.path.normpath(exp_folder)+'/*.jpg')
        for f in filenames:
            self.analyse_single_image(f, result_folder)


    #create table gathering segmentation info of all files found in folder
    def load_experiment(self, result_folder_exp):

        #newpath = result_folder+'/'+os.path.basename(os.path.normpath(folder))
        filenames = glob.glob(result_folder_exp+'/*.pkl')

        all_regions = []
        for f in filenames:
            all_regions.append(pd.read_pickle(f))

        ecc = pd.concat(all_regions,sort=False)

        return ecc

    def gauss_fit(self, x, a, x0, s): 
        return a*np.exp(-0.5*((x-x0)/s)**2)

    def gauss_fit2(self, x, a, x0, s, a2, x02, s2): 
        return a*np.exp(-0.5*((x-x0)/s)**2) + a2*np.exp(-0.5*((x-x02)/s2)**2)

    def normal_fit(self, x, a, x0, s): 
        return (a/(s*(2*np.pi)**0.5))*np.exp(-0.5*((x-x0)/s)**2)

    def split_categories(self, result_folder_exp):

        #recover all the spore properties for all images
        ecc_table_or = self.load_experiment(result_folder_exp)
        ecc_table = ecc_table_or.copy()
        
        #remove too small and too large regions
        ecc_table = ecc_table_or[ecc_table_or.area > self.min_area]
        ecc_table = ecc_table[ecc_table.area < self.max_area]

        #reshape eccentricity array for sklearn
        X = np.reshape(ecc_table.ecc.values,(-1,1))
        
        #if no threshold is provided, classify using EM
        if self.threshold is None:

            #create EM object. Initialization is important to ensure the two classes don't overlap
            GM = mixture.GaussianMixture(n_components=2,means_init = np.reshape([0.5,0.9],(-1,1)))
            
            #classifiy the data
            GM.fit(X)

            #check wich class correspond to round cells. ind2 is always the round class
            if GM.means_[0]>GM.means_[1]:
                ind1,ind2 = 0, 1
            else:
                ind1,ind2 = 1, 0

            #calculate the fraction of round cells
            frac_round = len(X[GM.predict(X)==ind2])/len(X)

            #find threshold by finding the first category change in a fine-grained eccentricity range
            threshold = np.arange(0,1,0.001)[np.argwhere(np.abs(np.diff(GM.predict(np.reshape(np.arange(0,1,0.001),(-1,1)))))>0)[0]][0]

            #create a list of categories and add to dataframe
            category = (GM.predict(np.reshape(ecc_table_or.ecc.values,(-1,1)))==ind2).astype(int)
            ecc_table_or['roundcat'] = category
        else:
            #if a fixed thresdhold is provieded, just calculate the fraction of cells below threshold
            threshold = self.threshold
            ecc_table['roundcat'] = ecc_table.ecc.apply(lambda x: 1 if x<threshold else 0)
            ecc_table_or['roundcat'] = ecc_table_or.ecc.apply(lambda x: 1 if x<threshold else 0)
            frac_round = np.sum(ecc_table['roundcat'])/len(ecc_table)
        
        #group dataframe by filename to re-export a summary file per image as csv
        grouped = ecc_table_or.groupby('filename',as_index=False)
        for indk, k in enumerate(list(grouped.groups.keys())):
            cur_group = grouped.get_group(k)
            cur_group.to_pickle(result_folder_exp+'/'+os.path.basename(cur_group.iloc[0].filename).split('.')[0]+'.pkl')
            cur_group[['area','ecc','roundcat']].to_csv(result_folder_exp+'/'+os.path.basename(cur_group.iloc[0].filename).split('.')[0]+'.csv', mode = 'w', index = False)

        #remove too small and too large spores from original dataframe and export as csv
        ecc_table_or = ecc_table_or[ecc_table_or.area >= self.min_area]
        ecc_table_or = ecc_table_or[ecc_table_or.area <= self.max_area]
        ecc_table_or[['area','ecc','roundcat']].to_csv(result_folder_exp+'/'+
                                                       os.path.basename(os.path.normpath(result_folder_exp))+'_summary.csv',
                                                       index = False)


        #create a histogram figure
        hist_val, xdata = np.histogram(X,bins = np.arange(0,1,0.005),density=True)
        xdata = np.array([0.5*(xdata[x]+xdata[x+1]) for x in range(len(xdata)-1)])

        fig, ax = plt.subplots()
        plt.bar(x=xdata, height=hist_val, width=xdata[1]-xdata[0],color = 'gray',label='Data')
        #plt.plot(xdata, self.normal_fit(xdata,GM.weights_[0], GM.means_[0,0], GM.covariances_[0,0]**0.5)+
        #         self.normal_fit(xdata,GM.weights_[1], GM.means_[1,0], GM.covariances_[1,0]**0.5),'k',label='Double gauss')

        #if EM is performed, show gaussian fits
        if self.threshold is None:
            plt.plot(xdata,self.normal_fit(xdata,GM.weights_[ind1], GM.means_[ind1,0], GM.covariances_[ind1,0,0]**0.5),
                 'b',linewidth = 2, label='Elongated spores')
            plt.plot(xdata,self.normal_fit(xdata,GM.weights_[ind2], GM.means_[ind2,0], GM.covariances_[ind2,0,0]**0.5),
                 'r',linewidth = 2, label='Round spores')
        plt.plot([threshold,threshold],[0,np.max(hist_val)],'k--',linewidth = 2,label='Threshold')

        ax.set_xlabel('Eccentricity',fontdict=font)
        ax.set_xlim([0.2, 1.0])

        if self.show_legend:
            ax.legend()
        if self.show_title:
            ax.set_title(os.path.basename(os.path.normpath(result_folder_exp))+', Round frac: '+str(np.around(frac_round,decimals=2)))
        if self.show_output:
            plt.show()
        fig.savefig(result_folder_exp+'/'+os.path.basename(os.path.normpath(result_folder_exp))+'_summary.png')
        plt.close(fig)
        pd.DataFrame({'name': [os.path.basename(os.path.normpath(result_folder_exp))],'fraction':[np.around(frac_round,decimals=2)]}).to_csv(result_folder_exp+'/'+os.path.basename(os.path.normpath(result_folder_exp))+'.csv',index = False, header=False)


    def plot_image_categories(self, exp_folder, result_folder):

        result_folder_exp = self.path_to_analysis(exp_folder, result_folder)

        np.random.seed(1)
        cmap = matplotlib.colors.ListedColormap (np.random.rand(256,3))

        #im_files = glob.glob(os.path.normpath(exp_folder)+'/*.jpg')
        ecc_table = self.load_experiment(result_folder_exp)
        im_files = ecc_table.filename.unique()

        for f in im_files:
            image = skimage.io.imread(f)[:,:,0]
            cur_spores = ecc_table[ecc_table.filename == f]
            empty_im = np.zeros(image.shape)
            for x in cur_spores.index:
                if (cur_spores.loc[x].area>self.min_area) and (cur_spores.loc[x].area<self.max_area):
                    if (cur_spores.loc[x].roundcat==1):
                        empty_im[cur_spores.loc[x].coords[:,0],cur_spores.loc[x].coords[:,1]]=1
                    else:
                        empty_im[cur_spores.loc[x].coords[:,0],cur_spores.loc[x].coords[:,1]]=2

            empty_im = empty_im.astype(float)
            empty_im[empty_im==0]=np.nan

            fig, ax = plt.subplots(figsize=(10,10))
            plt.imshow(image,cmap = 'gray')
            plt.imshow(empty_im,cmap = cmap,alpha = 0.9,vmin=0,vmax = 14)
            ax.set_axis_off()
            plt.tight_layout()
            if self.show_output:
                plt.show()
            fig.savefig(result_folder_exp+'/'+os.path.basename(f).split('.')[0]+'_classes.png', dpi = 300)
            plt.close(fig)
