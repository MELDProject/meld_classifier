from setuptools import setup, find_packages

setup(
    name="meld_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=["nibabel", "h5py", "pillow", "tensorflow", "pandas", "matplotlib"],
    package_dir={"meld_classifier": "meld_classifier"},
)
