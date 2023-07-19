"""
Name: utils.py
Author: Michal Mikeska
Last update: 25.5.2023
"""
import SimpleITK as sitk
import nibabel as nib
import os
import cv2


class Utils:
    @staticmethod
    def nib_load(path2file):
        """
        Load a NIfTI image from the specified file.

        Args:
            path2file (str): The path to the NIfTI file.

        Returns:
            nib_img: The loaded NIfTI image.
        """
        try:
            nib_img = nib.load(path2file)
            return nib_img
        except FileNotFoundError:
            print("File not found. Please enter a valid file path.")

    @staticmethod
    def nib_save(path2file, nib_img):
        """
        Save a NIfTI image to the specified file.

        Args:
            path2file (str): The path to the file where the NIfTI image will be saved.
            nib_img: The NIfTI image to be saved.

        Returns:
            None
        """
        nib.save(nib_img, path2file)

    @staticmethod
    def stik_load(path2file):
        """
        Load a SimpleITK image from the specified file.

        Args:
            path2file (str): The path to the SimpleITK file.

        Returns:
            sikt_img: The loaded SimpleITK image.
        """
        try:
            sikt_img = sitk.ReadImage(path2file, sitk.sitkFloat32)
            return sikt_img
        except FileNotFoundError:
            print("File not found. Please enter a valid file path.")

    @staticmethod
    def create_folder_if_not_exists(path2folder):
        """
        Create a folder if it doesn't exist.

        Args:
            path2folder (str): Path to the folder to be created.

        Returns:
            None
        """
        if not os.path.exists(path2folder):
            os.makedirs(path2folder)

    @staticmethod
    def plot_examples_from_np(original_arr, changed_arr, mask_arr, path2output_folder):
        """
        Plot and save example images from numpy arrays.

        Args:
            original_arr: The original image array.
            changed_arr: The changed image array.
            mask_arr: The mask image array.
            path2output_folder (str): Path to the folder where the output images will be saved.

        Returns:
            None
        """
        try:
            dims = original_arr.shape
            for idx in range(dims[0]):
                img1_1 = cv2.cvtColor(original_arr[idx], cv2.COLOR_GRAY2RGB)
                img1_2 = cv2.cvtColor(original_arr[:, idx, :], cv2.COLOR_GRAY2RGB)
                img1_3 = cv2.cvtColor(original_arr[..., idx], cv2.COLOR_GRAY2RGB)
                img2_1 = cv2.cvtColor(changed_arr[idx], cv2.COLOR_GRAY2RGB)
                img2_2 = cv2.cvtColor(changed_arr[:, idx, :], cv2.COLOR_GRAY2RGB)
                img2_3 = cv2.cvtColor(changed_arr[..., idx], cv2.COLOR_GRAY2RGB)
                img3_1 = cv2.cvtColor(mask_arr[idx] * 255, cv2.COLOR_GRAY2RGB)
                img3_2 = cv2.cvtColor(mask_arr[:, idx, :] * 255, cv2.COLOR_GRAY2RGB)
                img3_3 = cv2.cvtColor(mask_arr[..., idx] * 255, cv2.COLOR_GRAY2RGB)

                cv2.putText(img1_1, "raw scan", (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                cv2.putText(img2_1, "new scan", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                cv2.putText(img3_1, "mask", (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

                img1 = cv2.vconcat([img1_1, img1_2, img1_3])
                img2 = cv2.vconcat([img2_1, img2_2, img2_3])
                img3 = cv2.vconcat([img3_1, img3_2, img3_3])

                img = [img1, img2, img3]
                img = cv2.hconcat(img)
                Utils.create_folder_if_not_exists(path2output_folder)
                cv2.imwrite(f'{path2output_folder}/{idx}.png', img)
        except FileNotFoundError:
            print("Something went wrong during plotting the example images")
