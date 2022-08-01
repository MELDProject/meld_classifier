from setuptools import setup, find_packages

try:
    from meld_classifier import __author__, __maintainer__, __email__, __version__
except ImportError:
    __author__ = __maintainer__ = "MELD development team"
    __email__ = "meld.study@gmail.com"
    __version__ = "0.1.0"

setup(
    name="meld_classifier",
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description="Neural network lesion classifier for the MELD project.",
    license="MIT",
    packages=find_packages(),
    install_requires=["nibabel", "h5py", "pillow", "tensorflow", "pandas", "matplotlib"],
    package_dir={"meld_classifier": "meld_classifier"},
)
