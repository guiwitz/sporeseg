import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.morphology
import skimage.segmentation
import skimage.io
from sklearn import mixture
from scipy.ndimage.morphology import binary_fill_holes
#from scipy.optimize import curve_fit
import matplotlib
cmap = matplotlib.colors.ListedColormap (np.random.rand(256,3))
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }


class Spore:
    
    def __init__(self, show_output = False, min_area = 250, max_area = 1500, bin_width = 0.01,
                 show_title = True, show_legend = True, threshold = None, convexity = 0.9):

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
        bin_width : float
            bin width of eccentricity histogram
        threshold : float
            fixed threshold to use for splitting
        convexity : float (0-1)
            threshold for convexity. Convex object have a value of 1
        """

        self.show_output = show_output
        self.show_title = show_title
        self.show_legend = show_legend
        self.min_area = min_area
        self.max_area = max_area
        self.bin_width = bin_width
        self.threshold = threshold
        self.convexity = convexity
        

    def path_to_analysis(self, data_path, result_folder):
        """Returns the path to the result folder of an analysis
    
        Parameters
        ----------
        data_path : str
            path to data
        result_folder : str
            main folder of results

        Returns
        -------
        result_folder_exp: str 
            folder of results for a given dataset
        """

        if os.path.isfile(data_path):
            result_folder_exp = result_folder+'/'+os.path.basename(os.path.dirname(data_path))
        else:
            result_folder_exp = os.path.normpath(result_folder)+'/'+os.path.basename(os.path.normpath(data_path))

        if not os.path.isdir(result_folder_exp):
            os.makedirs(result_folder_exp, exist_ok=True)

        return result_folder_exp


    def analyse_single_image(self, image_path, result_folder):
        """Segment a single image. Saves segmentation images and
        segmentation results as pkl and csv
    
        Parameters
        ----------
        image_path : str
            path to image
        result_folder : str
            main folder of results

        Returns
        -------
        
        """
        
        regions, image, image_seg = self.find_spores(image_path)
        fig = self.plot_segmentation(image, image_seg)

        result_folder_exp = self.path_to_analysis(image_path, result_folder)

        regions.to_pickle(result_folder_exp+'/'+os.path.basename(image_path).split('.')[0]+'.pkl')
        regions[['area','convex_area','ecc']].to_csv(result_folder_exp+'/'+os.path.basename(image_path).split('.')[0]+'.csv',index = False, float_format='%.5f')
        fig.savefig(result_folder_exp+'/'+os.path.basename(image_path).split('.')[0]+'_seg.png', dpi = image_seg.shape[0])
        plt.close(fig)
        #return regions, image, image_seg

        
    def find_spores(self, image_path):
        """Segmentation of an image.
    
        Parameters
        ----------
        image_path : str
            path to image

        Returns
        -------
        regions_prop: Dataframe 
            Pandas dataframe with information on segmented spores
        raw_im: numpy array
            median filtered image
        image_mask: numpy array
            binary mask of spores
        """

        #get a binary segmentation
        raw_im, image_mask = self.segmentation(image_path)
        
        #measure region properties and keep area, eccentricity and coords
        regions = skimage.measure.regionprops(skimage.morphology.label(image_mask),coordinates='rc')

        #collect relevant information
        regions_prop = pd.DataFrame({'area':[x.area for x in regions],'convex_area':[x.convex_area for x in regions],
                                     'ecc':[x.eccentricity for x in regions],
                                    'coords': [x.coords for x in regions]})
        regions_prop['filename'] = image_path

        return regions_prop, raw_im, image_mask

    def segmentation_old(self, image_path):
        """Creation of binary mask of spores.
    
        Parameters
        ----------
        image_path : str
            path to image

        Returns
        -------
        raw_im: numpy array
            median filtered image
        snake_mask: numpy array
            binary mask of spores
        """

        #import image
        image = skimage.io.imread(image_path)[:,:,0]
        #do a median filtering to remove outliers
        raw_image = skimage.filters.median(image,selem=skimage.morphology.disk(10))
        #do a large scale background subtraction by using a very large gaussian
        flatten = raw_image-skimage.filters.gaussian(raw_image,sigma=100,preserve_range=True)
        
        #calculate a binary mask using automatic Li thresholding
        #mask = flatten<skimage.filters.threshold_li(flatten)
        
        #calcualte a binary mask by using the background distribution (asssumes low density of spores)
        mask = flatten < np.mean(flatten)-2*np.std(flatten)
        
        #remove objects touching the border
        mask = skimage.segmentation.clear_border(mask,buffer_size = 10)
        mask_label = skimage.morphology.label(mask)
        
        reg_spores = skimage.measure.regionprops(skimage.morphology.label(mask), intensity_image=raw_image)
        
        #use an active contour on each element to make segmentation more accurate
        #create again a mask using those refined contours
        snake_mask = np.zeros(image.shape)

        #plt.figure(figsize=(20,20))
        #plt.imshow(image,cmap='gray')
        for i in range(len(reg_spores)):
            bbox = reg_spores[i].bbox
            small_im = skimage.filters.median(image[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11], skimage.morphology.disk(3))
            #small_im = raw_image[bbox[0]-5:bbox[2]+6,bbox[1]-5:bbox[3]+6]
            small_mask = mask_label[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11] == reg_spores[i].label
            small_mask = skimage.morphology.binary_dilation(small_mask, skimage.morphology.disk(2))

            contour = skimage.measure.find_contours(small_mask, level = 0.8)
            snake = skimage.segmentation.active_contour(small_im.astype(float),snake= np.fliplr(contour[0]), w_edge=0.005, w_line=0,alpha = 0.1, beta=0.1)#,max_iterations=2)
            #plt.plot(contour[0][:,1]+bbox[1]-10,contour[0][:,0]+bbox[0]-10,'b-')
            #plt.plot(snake[:,0]+bbox[1]-10,snake[:,1]+bbox[0]-10,'r-')

            rr, cc = skimage.draw.polygon(snake[:,1]+bbox[0]-10,
                                              snake[:,0]+bbox[1]-10, snake_mask.shape)
            snake_mask[rr,cc] = True
        #plt.show()

        return raw_image, snake_mask
    
    def segmentation_new(self, image_path):
        """Alternative solution for creation of binary mask of spores.
    
        Parameters
        ----------
        image_path : str
            path to image

        Returns
        -------
        raw_im: numpy array
            median filtered image
        snake_mask: numpy array
            binary mask of spores
        """
        
        #import image
        image = skimage.io.imread(image_path)[:,:,0]
        #do a median filtering to remove outliers. Done one rescaled image for speed
        raw_image = skimage.filters.median(image[::2,::2],selem=skimage.morphology.disk(5))
        raw_image = skimage.transform.resize(raw_image, image.shape, order = 1, preserve_range=True).astype(np.uint8)

        border_mask = skimage.morphology.dilation(
            skimage.morphology.thin(
                skimage.filters.rank.gradient(
                    raw_image,skimage.
                    morphology.disk(2))>10),
            skimage.morphology.disk(1))

        inv_border_label = skimage.morphology.label(~border_mask)

        regions = skimage.measure.regionprops(inv_border_label)
        indices = np.array([0]+[x.label if (x.area>self.min_area)&(x.area<self.max_area) else 0 for x in regions])
        mask = indices[skimage.morphology.label(inv_border_label)]>0
        mask = skimage.segmentation.clear_border(mask,buffer_size = 10)
        mask_label = skimage.morphology.label(mask)

        #analyze the binary image
        reg_spores = skimage.measure.regionprops(skimage.morphology.label(mask), intensity_image=raw_image)

        #find the peak of average intensity within objects and create a threshold
        int_values = np.array([x.mean_intensity for x in reg_spores])
        hist_val, hist_x = np.histogram(int_values,np.arange(0,255,10))
        threshold = hist_x[np.argmax(hist_val)] + 40

        #use an active contour on each element to make segmentation more accurate
        #create again a mask using those refined contours
        snake_mask = np.zeros(image.shape)

        #plt.figure(figsize=(20,20))
        #plt.imshow(image,cmap='gray')
        for i in range(len(reg_spores)):
            bbox = reg_spores[i].bbox
            small_im = skimage.filters.median(image[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11], skimage.morphology.disk(3))
            #small_im = raw_image[bbox[0]-5:bbox[2]+6,bbox[1]-5:bbox[3]+6]
            small_mask = mask_label[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11] == reg_spores[i].label
            small_mask = skimage.morphology.binary_dilation(small_mask, skimage.morphology.disk(2))

            if reg_spores[i].mean_intensity<threshold:
                contour = skimage.measure.find_contours(small_mask, level = 0.8)
                snake = skimage.segmentation.active_contour(small_im.astype(float),snake= np.fliplr(contour[0]), w_edge=0.005, w_line=0,alpha = 0.1, beta=0.1)#,max_iterations=2)
                #plt.plot(contour[0][:,1]+bbox[1]-10,contour[0][:,0]+bbox[0]-10,'b-')
                #plt.plot(snake[:,0]+bbox[1]-10,snake[:,1]+bbox[0]-10,'r-')

                rr, cc = skimage.draw.polygon(snake[:,1]+bbox[0]-10,
                                              snake[:,0]+bbox[1]-10, snake_mask.shape)
                snake_mask[rr,cc] = True
        #plt.show()

        return image, snake_mask
    
    def segmentation(self, image_path):
        """Alternative solution for creation of binary mask of spores.
    
        Parameters
        ----------
        image_path : str
            path to image

        Returns
        -------
        raw_im: numpy array
            median filtered image
        newmask: numpy array
            binary mask of spores
        """
        image = skimage.io.imread(image_path)[:,:,0]
        raw_image = skimage.filters.median(image[::2,::2],selem=skimage.morphology.disk(5))
        raw_image = skimage.transform.resize(raw_image, image.shape, order = 1, preserve_range=True).astype(np.uint8)

        border_mask = skimage.morphology.dilation(
                    skimage.morphology.thin(
                        skimage.filters.rank.gradient(
                            raw_image,skimage.
                            morphology.disk(2))>10),
                    skimage.morphology.disk(1))

        filled = binary_fill_holes(border_mask)^border_mask

        filled = skimage.segmentation.clear_border(filled,buffer_size = 10)

        filled_lab = skimage.morphology.label(filled)

        reg_spores = skimage.measure.regionprops(filled_lab)

        newmask = np.zeros(filled.shape)
        for i in range(len(reg_spores)):
            if reg_spores[i].area >self.min_area:
                bbox = reg_spores[i].bbox
                #small_im = skimage.filters.median(image[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11], skimage.morphology.disk(3))
                small_im = image[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11]
                threshold = skimage.filters.threshold_li(small_im)
                newmask[bbox[0]-10:bbox[2]+11,bbox[1]-10:bbox[3]+11] = binary_fill_holes(small_im<threshold)
                
        return image, newmask



    def plot_segmentation(self, image, image_seg):
        """Plot and save the superposition of an image and its binary
        segmentation mask
    
        Parameters
        ----------
        image : numpy array
            intensity image
        image_seg : numpy array
            binary mask array
            

        Returns
        -------
        fig: matplotlib figure
            
        """
        
        image_seg = image_seg.astype(float)
        image_seg[image_seg == 0] = np.nan
        
        sizes = image_seg.shape
        height = float(sizes[0])
        width = float(sizes[1])
        
        fig = plt.figure()
        fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
    
        plt.imshow(image,cmap = 'gray')
        plt.imshow(image_seg,cmap = 'Reds',alpha = 0.7,vmin=0,vmax = 1.5)
        if self.show_output:
            plt.show()
        return fig
    

    #segment all images stored in folder. Save result in result_folder
    def analyse_spore_folder(self, exp_folder, result_folder):
        """Run segmentation on all images of a folder
    
        Parameters
        ----------
        exp_folder : str
            path to folder with images
        result_folder : str
            folder where to save results
            

        Returns
        -------
        
        """
        
        filenames = glob.glob(os.path.normpath(exp_folder)+'/*.jpg')
        for f in filenames:
            self.analyse_single_image(f, result_folder)


    #create table gathering segmentation info of all files found in folder
    def load_experiment(self, result_folder_exp):
        """Load all segmentation data (pkl files) of a folder.
    
        Parameters
        ----------
        result_folder_exp : str
            path to folder with results
            

        Returns
        -------
        ecc: Dataframe
            dataframe with data of all images
            
        """

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
        """Given a segmentation dataset split the results
        into two categories based on eccentricity. Results are saved 
        in the form of a csv file. The threshold between categories can
        be calcualted using a gaussian mixture or it can be manually set
        if self.threshold has a value.
    
        Parameters
        ----------
        result_folder_exp : str
            path to folder with results
            

        Returns
        -------
            
        """

        #recover all the spore properties for all images
        ecc_table_or = self.load_experiment(result_folder_exp)
        ecc_table = ecc_table_or.copy()
                
        #remove too small and too large regions
        ecc_table = ecc_table_or[ecc_table_or.area > self.min_area]
        ecc_table = ecc_table[ecc_table.area < self.max_area]
        
        #keep only convex objects
        ecc_table =  ecc_table[ecc_table.area/ecc_table.convex_area > self.convexity]
        #indices = np.array([0]+[x.label if x.area/x.convex_area>self.convexity else 0 for x in regions])
        #image_mask = indices[skimage.morphology.label(image_mask)]>0
        #regions = skimage.measure.regionprops(skimage.morphology.label(image_mask),coordinates='rc')

        #reshape eccentricity array for sklearn
        X = np.reshape(ecc_table.ecc.values,(-1,1))
        
        #if no threshold is provided, classify using EM
        if self.threshold is None:

            #create EM object. Initialization is important to ensure the two classes don't overlap
            GM = mixture.GaussianMixture(n_components=2,means_init = np.reshape([0.5,0.95],(-1,1)))
            
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
            cur_group[['area','convex_area','ecc','roundcat']].to_csv(result_folder_exp+'/'+os.path.basename(cur_group.iloc[0].filename).split('.')[0]+'.csv', mode = 'w', index = False, float_format='%.5f')

        #remove too small, too large spores and non-convex spores from original dataframe and export as csv
        ecc_table_or = ecc_table_or[ecc_table_or.area >= self.min_area]
        ecc_table_or = ecc_table_or[ecc_table_or.area <= self.max_area]
        ecc_table_or =  ecc_table_or[ecc_table_or.area/ecc_table_or.convex_area > self.convexity]
        ecc_table_or.filename = ecc_table_or.filename.apply(lambda x: os.path.basename(x))
        ecc_table_or[['filename','area','ecc','roundcat']].to_csv(result_folder_exp+'/'+
                                                       os.path.basename(os.path.normpath(result_folder_exp))+'_summary.csv',
                                                       index = False, header=['filename','area','eccentricity','round'], float_format='%.5f')


        #create a histogram figure
        hist_val, xdata = np.histogram(X,bins = np.arange(0,1,self.bin_width),density=True)
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
        """Plot a superposition of images and their segmentation
        with the two categories colored differently.
    
        Parameters
        ----------
        exp_folder : str
            path to folder with images
        result_folder : str
            folder where to save results
            

        Returns
        -------
        
        """

        result_folder_exp = self.path_to_analysis(exp_folder, result_folder)

        colmat = np.zeros((2,3)).astype(float)
        colmat[0,:] = [1,1,0]
        colmat[1,:] = [1,0,1]
        cmap = matplotlib.colors.ListedColormap(colmat)

        #im_files = glob.glob(os.path.normpath(exp_folder)+'/*.jpg')
        ecc_table = self.load_experiment(result_folder_exp)
        im_files = ecc_table.filename.unique()

        for f in im_files:
            image = skimage.io.imread(f)[:,:,0]
            cur_spores = ecc_table[ecc_table.filename == f]
            empty_im = np.zeros(image.shape)
            for x in cur_spores.index:
                if (cur_spores.loc[x].area>self.min_area) and (cur_spores.loc[x].area<self.max_area) \
                and (cur_spores.loc[x].area/cur_spores.loc[x].convex_area > self.convexity):
                    if (cur_spores.loc[x].roundcat==1):
                        empty_im[cur_spores.loc[x].coords[:,0],cur_spores.loc[x].coords[:,1]]=1
                    else:
                        empty_im[cur_spores.loc[x].coords[:,0],cur_spores.loc[x].coords[:,1]]=2

            empty_im = empty_im.astype(float)
            empty_im[empty_im==0]=np.nan
            
            sizes = image.shape
            height = float(sizes[0])
            width = float(sizes[1])

            fig = plt.figure()
            fig.set_size_inches(width/height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            plt.imshow(image,cmap = 'gray')
            plt.imshow(empty_im,cmap = cmap,alpha = 0.9)#,vmin=0,vmax = 14)
            if self.show_output:
                plt.show()
            fig.savefig(result_folder_exp+'/'+os.path.basename(f).split('.')[0]+'_classes.png', dpi = height)
            plt.close(fig)
