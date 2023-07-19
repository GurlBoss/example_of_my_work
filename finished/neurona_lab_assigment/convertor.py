"""
Name: convertor.py
Author: Michal Mikeska
Last update: 25.5.2023
"""


import dicom2nifti as d2n
import pydicom
import glob
import os

cwd = os.getcwd()


def get_case_name(path2dicom_folder):
    """Get case (patient_id and modality) from the DICOM files in one folder.

    Args:
        path2dicom_folder (str): Path to folder, where DICOM images are.
    Returns:
        str: The name of the case.
    """
    path2dicom_files = path2dicom_folder + "/*.dcm"
    path2dicom_sample = glob.glob(path2dicom_files)[0]
    ds = pydicom.dcmread(path2dicom_sample)
    new_format = "NIfTI"
    case_name = ds.PatientID + "_" + ds.Modality + "_" + new_format
    return case_name


def dicom2nifti(path2dicom_folder):
    """Convert DICOMS images in one nifti file.

    Args:
        path2dicom_folder (str): Path to folder, where DICOM images are.
    Returns:
        None
    """
    nifti_dirname = get_case_name(path2dicom_folder)
    if not os.path.exists(nifti_dirname):
        os.makedirs(nifti_dirname)
    path2nifti = cwd + "/" + nifti_dirname + "/" + nifti_dirname + ".nii.gz"
    d2n.dicom_series_to_nifti(path2dicom_folder, path2nifti, reorient_nifti=True)


def main():
    """
    Load dicom imgs and change them into one nifti file.
    """
    path2files = cwd + "/AD01_MR_DICOM"
    dicom2nifti(path2files)


if __name__ == '__main__':
    main()
