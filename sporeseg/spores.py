from pathlib import Path
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.morphology
import skimage.segmentation
import skimage.io
from sklearn import mixture
from scipy.ndimage.morphology import binary_fill_holes

import matplotlib

colmat = np.zeros((2, 3)).astype(float)
colmat[0, :] = [1, 1, 0]
colmat[1, :] = [1, 0, 1]
cmap = matplotlib.colors.ListedColormap(colmat)

font = {
    "family": "sans-serif",
    "color": "black",
    "weight": "normal",
    "size": 16,
}

class Spore:
    """
    Class defining a spore segmentation and analysis experiment.

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

    def __init__(
        self,
        show_output=False,
        min_area=250,
        max_area=1500,
        bin_width=0.01,
        show_title=True,
        show_legend=True,
        threshold=None,
        convexity=0.9,
    ):

        self.show_output = show_output
        self.show_title = show_title
        self.show_legend = show_legend
        self.min_area = min_area
        self.max_area = max_area
        self.bin_width = bin_width
        self.threshold = threshold
        self.convexity = convexity

    def path_to_analysis(self, data_path, result_folder):
        """
        Returns the path to the result folder of an analysis. If data_path
        points to a file, the folder is results_folder + parent folder of file. If
        data_path is a folder, the folder is results_folder + folder. If data_path
        is just a name, the folder is results_folder + name.

        Parameters
        ----------
        data_path : str or Path object
            path to file, folder or name
        result_folder : str
            main folder of results

        Returns
        -------
        result_folder_exp: str 
            folder of results for a given dataset
        """

        data_path = Path(data_path)
        result_folder = Path(result_folder)

        if data_path.is_file():
            result_folder_exp = result_folder.joinpath(data_path.parent.stem)
        else:
            result_folder_exp = result_folder.joinpath(data_path.stem)        

        if not os.path.isdir(result_folder_exp):
            os.makedirs(result_folder_exp, exist_ok=True)

        return result_folder_exp

    def analyse_single_image(self, image_path, result_folder, save_name=None):
        """
        Segment a single image. Saves segmentation images and
        segmentation results as pkl and csv

        Parameters
        ----------
        image_path : str or ndarray
            path to image
        result_folder : str
            main folder of results
        save_name: str
            name to use for saving the analysis. If none and image_path
            is a str, the image name is used, otherwise "image" is used

        Returns
        -------

        """

        # image loading
        if isinstance(image_path, np.ndarray):
            if save_name is None:
                save_name = 'image'
        else:
            image = skimage.io.imread(image_path)[:, :, 0]
            if save_name is None:
                image_path = Path(image_path)
                save_name = image_path.stem

        regions, image, image_seg = self.find_spores(image_path)
        fig = self.plot_segmentation(image, image_seg)

        result_folder_exp = self.path_to_analysis(save_name, result_folder)

        regions.to_pickle(result_folder_exp.joinpath(save_name+'.pkl'))
        
        regions[["area", "convex_area", "ecc", "centroid_x", "centroid_y"]].to_csv(
            result_folder_exp.joinpath(save_name+'.csv'),
            index=False,
            float_format="%.5f",
        )
        fig.savefig(
            result_folder_exp.joinpath(save_name+'_seg.png'), dpi=image_seg.shape[0],
        )
        plt.close(fig)

    def find_spores(self, image_path, image_name=None):
        """
        Segmentation of an image.

        Parameters
        ----------
        image_path : str or ndarray
            path to image or image
        image_name: str
            if image_path is an ndarray, name of image for saving

        Returns
        -------
        regions_prop: Dataframe 
            Pandas dataframe with information on segmented spores
        raw_im: numpy array
            median filtered image
        image_mask: numpy array
            binary mask of spores
        """

        # get a binary segmentation
        raw_im, image_mask = self.segmentation(image_path)

        # measure region properties and keep area, eccentricity and coords
        regions = skimage.measure.regionprops(
            skimage.morphology.label(image_mask), coordinates="rc"
        )

        # collect relevant information
        regions_prop = pd.DataFrame(
            {
                "area": [x.area for x in regions],
                "convex_area": [x.convex_area for x in regions],
                "ecc": [x.eccentricity for x in regions],
                "coords": [x.coords for x in regions],
                "centroid_x": [x.centroid[0] for x in regions],
                "centroid_y": [x.centroid[1] for x in regions],
            }
        )

        if isinstance(image_path, np.ndarray):
            if image_name is None:
                image_name = 'image'
            regions_prop["filename"] = image_name
        else:
            regions_prop["filename"] = str(Path(image_path))

        return regions_prop, raw_im, image_mask

    def segmentation(self, image_path, gradient_threshold = 10):
        """
        Alternative solution for creation of binary mask of spores.

        Parameters
        ----------
        image_path : str or ndarray
            path to image or image

        Returns
        -------
        raw_im: numpy array
            median filtered image
        newmask: numpy array
            binary mask of spores
        """

        # image loading
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image = skimage.io.imread(image_path)[:, :, 0]            
        
        # median filtering
        raw_image = skimage.filters.median(
            image[::2, ::2], selem=skimage.morphology.disk(5)
        )
        # image upscaling
        raw_image = skimage.transform.resize(
            raw_image, image.shape, order=1, preserve_range=True
        ).astype(np.uint8)

        #obtain contours by thresholding the gradient image
        border_mask = skimage.morphology.dilation(
            skimage.morphology.thin(
                skimage.filters.rank.gradient(raw_image, skimage.morphology.disk(2))
                > gradient_threshold
            ),
            skimage.morphology.disk(1),
        )

        filled = binary_fill_holes(border_mask) ^ border_mask

        filled = skimage.segmentation.clear_border(filled, buffer_size=10)

        filled_lab = skimage.morphology.label(filled)

        reg_spores = skimage.measure.regionprops(filled_lab)

        newmask = np.zeros(filled.shape)
        for i in range(len(reg_spores)):
            if reg_spores[i].area > self.min_area:
                bbox = reg_spores[i].bbox

                small_im = image[
                    bbox[0] - 10 : bbox[2] + 11, bbox[1] - 10 : bbox[3] + 11
                ]
                threshold = skimage.filters.threshold_li(small_im)
                newmask[
                    bbox[0] - 10 : bbox[2] + 11, bbox[1] - 10 : bbox[3] + 11
                ] = binary_fill_holes(small_im < threshold)

        return image, newmask

    def plot_segmentation(self, image, image_seg, fig_scaling=1):
        """
        Plot and save the superposition of an image and its binary
        segmentation mask

        Parameters
        ----------
        image : numpy array
            intensity image
        image_seg : numpy array
            binary mask array
        fig_scaling: int
            scaling factor for displayed image


        Returns
        -------
        fig: matplotlib figure

        """
        
        image_seg = image_seg.astype(float)
        image_seg[image_seg == 0] = np.nan

        sizes = image_seg.shape
        height = float(sizes[0])
        width = float(sizes[1])

        factor = 1
        fig = plt.figure()
        fig.set_size_inches(width / height, fig_scaling, forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0*fig_scaling, 1.0*fig_scaling])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(image, cmap="gray")
        plt.imshow(image_seg, cmap="Reds", alpha=0.7, vmin=0, vmax=1.5)
        if self.show_output:
            plt.show()
        return fig

    def analyse_spore_folder(self, exp_folder, result_folder):
        """
        Run segmentation on all images of a folder

        Parameters
        ----------
        exp_folder : str
            path to folder with images
        result_folder : str
            folder where to save results


        Returns
        -------

        """

        filenames = Path(exp_folder).glob("*.jpg")
        for f in filenames:
            self.analyse_single_image(f, result_folder)

    def load_experiment(self, result_folder_exp):
        """
        Load all segmentation data (pkl files) of a folder.

        Parameters
        ----------
        result_folder_exp : str
            path to folder with results


        Returns
        -------
        ecc: Dataframe
            dataframe with data of all images

        """

        filenames = Path(result_folder_exp).glob("*.pkl")

        all_regions = []
        for f in filenames:
            all_regions.append(pd.read_pickle(f))

        ecc = pd.concat(all_regions, sort=False)

        return ecc

    def gauss_fit(self, x, a, x0, s):
        return a * np.exp(-0.5 * ((x - x0) / s) ** 2)

    def gauss_fit2(self, x, a, x0, s, a2, x02, s2):
        return a * np.exp(-0.5 * ((x - x0) / s) ** 2) + a2 * np.exp(
            -0.5 * ((x - x02) / s2) ** 2
        )

    def normal_fit(self, x, a, x0, s):
        return (a / (s * (2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - x0) / s) ** 2)

    def split_categories(self, result_folder_exp):
        """
        Given a segmentation dataset split the results
        into two categories based on eccentricity. Results are saved 
        in the form of a csv file. The threshold between categories can
        be calculated using a gaussian mixture or it can be manually set
        if self.threshold has a value.

        Parameters
        ----------
        result_folder_exp : str or Path object
            path to folder with results


        Returns
        -------

        """

        result_folder_exp = Path(result_folder_exp)

        # recover all the spore properties for all images
        ecc_table_or = self.load_experiment(result_folder_exp)
        ecc_table = ecc_table_or.copy()

        # remove too small and too large regions
        ecc_table = ecc_table_or[ecc_table_or.area > self.min_area]
        ecc_table = ecc_table[ecc_table.area < self.max_area]

        # keep only convex objects
        ecc_table = ecc_table[ecc_table.area / ecc_table.convex_area > self.convexity]
        
        # reshape eccentricity array for sklearn
        X = np.reshape(ecc_table.ecc.values, (-1, 1))

        # if no threshold is provided, classify using EM
        if self.threshold is None:

            # create EM object. Initialization is important to ensure the two classes don't overlap
            GM = mixture.GaussianMixture(
                n_components=2, means_init=np.reshape([0.5, 0.95], (-1, 1))
            )

            # classifiy the data
            GM.fit(X)

            # check wich class correspond to round cells. ind2 is always the round class
            if GM.means_[0] > GM.means_[1]:
                ind1, ind2 = 0, 1
            else:
                ind1, ind2 = 1, 0

            # calculate the fraction of round cells
            frac_round = len(X[GM.predict(X) == ind2]) / len(X)

            # find threshold by finding the first category change in a fine-grained eccentricity range
            threshold = np.arange(0, 1, 0.001)[
                np.argwhere(
                    np.abs(
                        np.diff(GM.predict(np.reshape(np.arange(0, 1, 0.001), (-1, 1))))
                    )
                    > 0
                )[0]
            ][0]

            # create a list of categories and add to dataframe
            category = (
                GM.predict(np.reshape(ecc_table_or.ecc.values, (-1, 1))) == ind2
            ).astype(int)
            ecc_table_or["roundcat"] = category
        else:
            # if a fixed thresdhold is provieded, just calculate the fraction of cells below threshold
            threshold = self.threshold
            ecc_table["roundcat"] = ecc_table.ecc.apply(
                lambda x: 1 if x < threshold else 0
            )
            ecc_table_or["roundcat"] = ecc_table_or.ecc.apply(
                lambda x: 1 if x < threshold else 0
            )
            frac_round = np.sum(ecc_table["roundcat"]) / len(ecc_table)

        # group dataframe by filename to re-export a summary file per image as csv
        grouped = ecc_table_or.groupby("filename", as_index=False)
        for indk, k in enumerate(list(grouped.groups.keys())):
            cur_group = grouped.get_group(k)
            cur_group.to_pickle(
                result_folder_exp.joinpath(Path(cur_group.iloc[0].filename).stem + ".pkl"
            ))
            cur_group[["centroid_x", "centroid_y", "area", "convex_area", "ecc", "roundcat"]].to_csv(
                result_folder_exp.joinpath(
                Path(cur_group.iloc[0].filename).stem + ".csv"),
                mode="w",
                index=False,
                float_format="%.5f",
            )

        # remove too small, too large spores and non-convex spores from original dataframe and export as csv
        ecc_table_or = ecc_table_or[ecc_table_or.area >= self.min_area]
        ecc_table_or = ecc_table_or[ecc_table_or.area <= self.max_area]
        ecc_table_or = ecc_table_or[
            ecc_table_or.area / ecc_table_or.convex_area > self.convexity
        ]
        ecc_table_or.filename = ecc_table_or.filename.apply(
            lambda x: os.path.basename(x)
        )
        ecc_table_or[["filename", "centroid_x", "centroid_y", "area", "ecc", "roundcat"]].to_csv(
            result_folder_exp.joinpath(Path(result_folder_exp.stem) + "_summary.csv"),
            index=False,
            header=["filename", "centroid_x", "centroid_y", "area", "eccentricity", "round"],
            float_format="%.5f",
        )

        # create a histogram figure
        hist_val, xdata = np.histogram(
            X, bins=np.arange(0, 1, self.bin_width), density=True
        )
        xdata = np.array(
            [0.5 * (xdata[x] + xdata[x + 1]) for x in range(len(xdata) - 1)]
        )

        fig, ax = plt.subplots()
        plt.bar(
            x=xdata,
            height=hist_val,
            width=xdata[1] - xdata[0],
            color="gray",
            label="Data",
        )
        # plt.plot(xdata, self.normal_fit(xdata,GM.weights_[0], GM.means_[0,0], GM.covariances_[0,0]**0.5)+
        #         self.normal_fit(xdata,GM.weights_[1], GM.means_[1,0], GM.covariances_[1,0]**0.5),'k',label='Double gauss')

        # if EM is performed, show gaussian fits
        if self.threshold is None:
            plt.plot(
                xdata,
                self.normal_fit(
                    xdata,
                    GM.weights_[ind1],
                    GM.means_[ind1, 0],
                    GM.covariances_[ind1, 0, 0] ** 0.5,
                ),
                "b",
                linewidth=2,
                label="Elongated spores",
            )
            plt.plot(
                xdata,
                self.normal_fit(
                    xdata,
                    GM.weights_[ind2],
                    GM.means_[ind2, 0],
                    GM.covariances_[ind2, 0, 0] ** 0.5,
                ),
                "r",
                linewidth=2,
                label="Round spores",
            )
        plt.plot(
            [threshold, threshold],
            [0, np.max(hist_val)],
            "k--",
            linewidth=2,
            label="Threshold",
        )

        ax.set_xlabel("Eccentricity", fontdict=font)
        ax.set_xlim([0.2, 1.0])

        if self.show_legend:
            ax.legend()
        if self.show_title:
            ax.set_title(
                result_folder_exp.stem
                + ", Round frac: "
                + str(np.around(frac_round, decimals=2))
            )
        if self.show_output:
            plt.show()
        fig.savefig(
            result_folder_exp.joinpath(result_folder_exp.stem + "_summary.png")
        )
        plt.close(fig)
        pd.DataFrame(
            {
                "name": [result_folder_exp.stem],
                "fraction": [np.around(frac_round, decimals=2)],
            }
        ).to_csv(
            result_folder_exp.joinpath(
            result_folder_exp.stem
            + ".csv"),
            index=False,
            header=False,
        )

    def plot_image_categories(self, exp_folder, result_folder):
        """
        Plot a superposition of images and their segmentation
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

        ecc_table = self.load_experiment(result_folder_exp)
        im_files = ecc_table.filename.unique()

        for f in im_files:
            image = skimage.io.imread(f)[:, :, 0]
            cur_spores = ecc_table[ecc_table.filename == f]
            empty_im = np.zeros(image.shape)
            for x in cur_spores.index:
                if (
                    (cur_spores.loc[x].area > self.min_area)
                    and (cur_spores.loc[x].area < self.max_area)
                    and (
                        cur_spores.loc[x].area / cur_spores.loc[x].convex_area
                        > self.convexity
                    )
                ):
                    if cur_spores.loc[x].roundcat == 1:
                        empty_im[
                            cur_spores.loc[x].coords[:, 0],
                            cur_spores.loc[x].coords[:, 1],
                        ] = 1
                    else:
                        empty_im[
                            cur_spores.loc[x].coords[:, 0],
                            cur_spores.loc[x].coords[:, 1],
                        ] = 2

            empty_im = empty_im.astype(float)
            empty_im[empty_im == 0] = np.nan

            sizes = image.shape
            height = float(sizes[0])
            width = float(sizes[1])

            fig = plt.figure()
            fig.set_size_inches(width / height, 1, forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)

            plt.imshow(image, cmap="gray")
            plt.imshow(empty_im, cmap=cmap, alpha=0.9, vmin=0, vmax=3)
            if self.show_output:
                plt.show()
            fig.savefig(
                result_folder_exp.joinpath(f.stem + "_classes.png"),
                dpi=height,
            )
            plt.close(fig)
