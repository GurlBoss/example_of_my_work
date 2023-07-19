"""
Name: resize_and_plot.py
Author: Michal Mikeska
Last update: 25.5.2023
"""
import matplotlib.pyplot as plt
import numpy as np
from nilearn.image import resample_img
from utils import Utils
from nilearn import plotting


class ResizeAndPlot:
    path2file = None
    nib_img = None
    img_data = None
    shape = None

    """
    Resize and Plot NIfTI Images.

    This class provides methods to load, resize, save, and plot NIfTI images.

    Class Variables:
        path2file (str): Path to the input NIfTI file.
        nib_img (Nifti1Image): Loaded NIfTI image.
        img_data (numpy.ndarray): Image data of the NIfTI image.
        shape (tuple): Shape of the NIfTI image data.
    """
    def load(self, path2nifti_file):
        """
        Load a NIfTI image from the specified file.

        This method sets the path to the NIfTI file, loads the image using Utils.nib_load,
        and updates the image information.

        Args:
            path2nifti_file (str): The path to the NIfTI file.

        Returns:
            None
        """
        self.path2file = path2nifti_file
        self.nib_img = Utils.nib_load(self.path2file)
        self.update_information()

    def resize(self, target_shape, scaling_factor=1):
        """
        Resize the NIfTI image to the specified target shape and scaling factor.

        Args:
            target_shape (tuple or list): The target shape of the resized image in (x, y, z) format.
            scaling_factor (float, optional): The scaling factor to apply during resizing. Defaults to 1.

        Returns:
            None
        """
        target_shape = np.array(target_shape)
        new_resolution = [scaling_factor, ] * 3
        new_affine = np.zeros((4, 4))
        new_affine[:3, :3] = np.diag(new_resolution)
        new_affine[:3, 3] = target_shape * new_resolution / 2. * -1
        new_affine[3, 3] = 1.
        self.nib_img = resample_img(self.nib_img, target_affine=new_affine, target_shape=target_shape,
                                    interpolation='nearest')
        self.update_information()

    def update_information(self):
        """
        Update the image information based on the current NIfTI image.

        This method retrieves the image data and updates the shape attribute of the object.

        Returns:
            None
        """
        self.img_data = self.nib_img.get_fdata()
        self.shape = self.img_data.shape

    def save(self,path2file = None):
        """
        Save the NIfTI image to a file.

        Args:
            path2file (str, optional): The path to the file where the NIfTI image will be saved.
                If not provided, the default path specified during object initialization will be used.

        Returns:
            None
        """
        if not path2file:
            path2file = self.path2file
        Utils.nib_save(path2file=path2file,nib_img=self.nib_img)

    def plot(self,direction,path2file):
        """
       Plot slices of the image in the specified direction and save the figure to a file.

       Args:
           direction (str): The direction in which to plot the slices. Possible values are 'x', 'y', or 'z'.
           path2file (str): The path to the file where the figure will be saved.

       Returns:
           None
           """
        n_rows, n_cols = 5, 5
        all_coords = plotting.find_cut_slices(
            self.nib_img, direction=direction, n_cuts=n_rows * n_cols
        )
        ax_size = 2.0
        margin = 0.05
        fig, all_axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_rows * ax_size, n_cols * ax_size),
            gridspec_kw={"hspace": margin, "wspace": margin},)
        left_right = True
        for coord, ax in zip(all_coords, all_axes.ravel()):
            display = plotting.plot_anat(
                self.nib_img,
                cut_coords=[coord],
                display_mode=direction,
                axes=ax,
                annotate=False
            )
            display.annotate(left_right=left_right)
            left_right = False
        plt.savefig(path2file, bbox_inches="tight")

    def plot_all(self,path2folder = "output/plots"):
        """
        Plot various figures that show different main views of the data.

        Args:
            path2folder (str): Path to the folder where the output plots will be saved. Defaults to "output/plots".

        Returns:
        None
        """
        Utils.create_folder_if_not_exists(path2folder)
        plotting.plot_anat(self.nib_img)
        plt.savefig(path2folder + "/basic_view.png", bbox_inches="tight")

        self.plot(direction="z",path2file=path2folder + "/transverse_plane.png")
        self.plot(direction="y", path2file=path2folder + "/median_plane.png")
        self.plot(direction="x", path2file=path2folder + "/frontal_plane.png")

def main():
    """
    Load image, plot the images for non-technical user and resize it.
    """
    tmp_scan = ResizeAndPlot()
    tmp_scan.load("AD01_MR_NIfTI/AD01_MR_NIfTI.nii.gz")
    tmp_scan.plot_all()
    tmp_scan.resize(target_shape=(256,256,256))
    tmp_scan.save("AD01_MR_NIfTI/AD01_MR_NIfTI_resized.nii.gz")


if __name__ == '__main__':
    main()
