"""
Name: n4_bias_correction.py
Author: Michal Mikeska
Last update: 25.5.2023
"""
import SimpleITK as sitk
from utils import Utils
import numpy as np


class N4BiasCorrection:
    """
     N4 Bias Field Correction.

     This class provides methods to perform N4 bias field correction on SimpleITK images.

     Class Variables:
         path2file (str): Path to the input file.
         sitk_img (SimpleITK.Image): Loaded SimpleITK image.
         shrink_factor (int): Shrink factor for image and mask resizing.
         mask (SimpleITK.Image): Binary mask generated from the image for correction.
         corrected_img (SimpleITK.Image): Corrected image after N4 bias field correction.
     """
    path2file = None
    sitk_img = None
    shrink_factor = 4
    mask = None
    corrected_img = None



    def perform_correction(self):
        """
        Perform bias field correction on the loaded SimpleITK image.

        This method applies N4 bias field correction to the loaded image using the SimpleITK library.
        It generates a binary mask using Otsu thresholding and applies the correction to obtain the corrected image.

        Raises:
            ValueError: If no valid SimpleITK image is found.

        Returns:
            None
        """
        if not self.sitk_img:
            raise ValueError("Invalid SimpleITK image. No image data found")

        raw_sikt_img = self.sitk_img
        input_image = raw_sikt_img

        mask_img = sitk.RescaleIntensity(raw_sikt_img, 0, 255)
        mask_img = sitk.OtsuThreshold(mask_img, 0, 1)
        self.mask = mask_img

        input_image = sitk.Shrink(input_image, [self.shrink_factor] * input_image.GetDimension())
        mask_img = sitk.Shrink(mask_img, [self.shrink_factor] * input_image.GetDimension())

        bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
        bias_corrector.Execute(input_image, mask_img)

        log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_sikt_img)
        corrected_image_full_resolution = raw_sikt_img / sitk.Exp(log_bias_field)

        self.corrected_img = corrected_image_full_resolution

    def load(self, path2file):
        """
        Load a SimpleITK image from the specified file.

        Args:
            path2file (str): The path to the file.

        Returns:
            None
        """
        self.path2file = path2file
        self.sitk_img = Utils.stik_load(path2file)

    def save_corrected(self, path2file=None):
        """
        Save the corrected image to a file.

        Args:
            path2file (str, optional): The path to the file where the corrected image will be saved.
                If not provided, the default path specified during object initialization will be used.

        Returns:
            None
        """
        if path2file is None:
            path2file = self.path2file
        sitk.WriteImage(self.corrected_img, path2file)

    def plot_examples(self, path2output_folder="output/n4_bias_example"):
        """
        Plot examples of the original image, corrected image, and mask.

        This method takes the original image, corrected image, and mask and plots them as examples
        in the specified output folder.

        Args:
            path2output_folder (str, optional): Path to the output folder where the examples will be saved.
                Defaults to "output/n4_bias_example".

        Returns:
            None
        """
        original_arr = np.uint8(sitk.GetArrayFromImage(self.sitk_img))
        corrected_arr = np.uint8(sitk.GetArrayFromImage(self.corrected_img))
        mask_arr = np.uint8(sitk.GetArrayFromImage(self.mask))
        Utils.plot_examples_from_np(
            original_arr=original_arr,
            mask_arr=mask_arr,
            changed_arr=corrected_arr,
            path2output_folder=path2output_folder
            )



def main():
    """
    Perform N4bias correction on nifti image.
    """
    path2file = "AD01_MR_NIfTI/AD01_MR_NIfTI_resized.nii.gz"
    correction = N4BiasCorrection()
    correction.load(path2file)
    correction.perform_correction()
    correction.plot_examples()
    correction.save_corrected("AD01_MR_NIfTI/AD01_MR_NIfTI_corrected.nii.gz")


if __name__ == '__main__':
    main()
