from setuptools import setup, find_packages

try:
    from meld_classifier import __author__, __maintainer__, __email__, __version__
except ImportError:
    __author__ = __maintainer__ = "MELD development team"
    __email__ = "meld.study@gmail.com"
    __version__ = "1.1.0"

setup(
    name="meld_classifier",
    version="1.1.0",
    packages=find_packages(),
    package_dir={"meld_classifier": "meld_classifier"},
)
