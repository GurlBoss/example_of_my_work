"""
Name: brain_extractor.py
Author: Michal Mikeska
Last update: 25.5.2023
"""

from utils import Utils
import nibabel as nib
import numpy as np
from nilearn import image
import skimage.morphology as morphology
from skimage.filters import threshold_otsu
from scipy import ndimage


class BrainExtractor:
    path2file = None
    mask_img = None
    brain_img = None
    nib_img = None
    """
        Brain Extractor for NIfTI Images.

        This class provides methods to load a NIfTI image, perform brain extraction, save the brain image,
        and plot example images for visualization.

        Class Variables:
            path2file (str): Path to the input NIfTI file.
            mask_img (Nifti1Image): Binary mask image representing the extracted brain region.
            brain_img (Nifti1Image): Image of the extracted brain region.
            nib_img (Nifti1Image): Loaded NIfTI image.

        """
    def load(self, path2file):
        """
        Load a NIfTI image from the specified file.

        Args:
            path2file (str): The path to the NIfTI file.

        Returns:
            None
        """
        self.path2file = path2file
        self.nib_img = Utils.nib_load(path2file)

    def save(self, path2file=None):
        """
        Save the brain image to a file.

        Args:
            path2file (str, optional): The path to the file where the brain image will be saved.
                If not provided, the default path specified during object initialization will be used.

        Returns:
            None
        """
        try:
            if not path2file:
                path2file = self.path2file
            Utils.nib_save(path2file, self.brain_img)
        except ValueError:
            print("Brain image is probably None. Make sure you performed extraction correctly")

    @staticmethod
    def multi_erosion(img, ero_number=1):
        """
        Perform multiple dilations on a binary image.

        Args:
            img (ndarray): The input binary image.
            dil_number (int, optional): The number of dilations to perform. Defaults to 1.

        Returns:
            ndarray: The dilated binary image.
        """
        for i in range(ero_number):
            img = morphology.erosion(img)
        return img

    @staticmethod
    def multi_dilation(img, dil_number=1):
        for i in range(dil_number):
            img = morphology.dilation(img)
        return img

    @staticmethod
    def select_n_largest_component(img, n=1):
        """
        Select the n largest connected component from a binary image.

        This static method takes a binary image as input and selects the n largest connected component
        based on its size. It labels the connected components, calculates their sizes, and returns
        a binary mask for the n largest component(s).

        Args:
            img (ndarray): The input binary image.
            n (int, optional): The number of N largest component to select. Defaults to 1.

        Returns:
            ndarray: A binary mask for the n largest component.
        """
        labeled_array, num_features = ndimage.label(img)
        component_sizes = np.bincount(labeled_array.ravel())
        largest_component_label = np.argsort(component_sizes)[::-1][n]
        largest_component_mask = labeled_array == largest_component_label
        return largest_component_mask

    def extract_brain_final(self):
        """
        Extract the brain from the loaded image using a series of processing steps.

        This method applies a series of processing steps to extract the brain from the loaded image.
        It calculates a threshold using Otsu's method, generates a binary mask based on the threshold,
        performs erosion, selects the second largest connected component, and applies dilation.
        The resulting binary mask is then used to extract the brain region from the original image.

        Returns:
            None
        """
        original_img_data = self.nib_img.get_fdata()
        threshold = threshold_otsu(original_img_data)
        binary_mask = np.where(original_img_data > threshold, 1, 0)
        binary_mask = self.multi_erosion(binary_mask, 3)
        binary_mask = self.select_n_largest_component(binary_mask, 2)
        binary_mask = self.multi_dilation(binary_mask, 4)
        self.mask_img = nib.Nifti1Image(binary_mask, self.nib_img.affine, self.nib_img.header)
        self.brain_img = image.math_img('img1 * img2', img1=self.mask_img, img2=self.nib_img)

    def plot_examples(self, path2output_folder="output/extract_example"):
        """
        Plot example images for visualization.

        This method plots example images, including the original image, brain-extracted image,
        and mask image, and saves the figures to the specified output folder.

        Args:
            path2output_folder (str, optional): The path to the output folder where the figures will be saved.
                Defaults to "output/extract_example".

        Returns:
            None
        """
        original_arr = np.uint8(self.nib_img.get_fdata())
        brain_arr = np.uint8(self.brain_img.get_fdata())
        mask_arr = np.uint8(self.mask_img.get_fdata())
        Utils.plot_examples_from_np(original_arr=original_arr,
                                    mask_arr=mask_arr
                                    , changed_arr=brain_arr,
                                    path2output_folder=path2output_folder)


def main():
    """
    Extract brain and show the results.
    """
    extractor = BrainExtractor()
    extractor.load("AD01_MR_NIfTI/AD01_MR_NIfTI_corrected.nii.gz")
    extractor.extract_brain_final()
    extractor.plot_examples()
    extractor.save("AD01_MR_NIfTI/AD01_MR_NIfTI_brain.nii.gz")


if __name__ == '__main__':
    main()
