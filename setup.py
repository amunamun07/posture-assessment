from distutils.core import setup
from setuptools import find_packages

setup(
    name="posture-assessment",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "loguru==0.6.0",
        "pyyaml==6.0",
        "opencv-python==4.7.0.68",
        "opencv-contrib-python==4.7.0.72",
        "pandas>=1.1.4",
        "torch>=1.7.0,!=1.12.0",
        "torchvision>=0.8.1,!=0.13.0",
        "requests>=2.23.0",
        "numpy>=1.18.5",
        "tqdm>=4.41.0",
        "scipy>=1.4.1",
        "matplotlib>=3.2.2",
        "seaborn>=0.11.0",
        "moviepy==1.0.3"
    ],
    include_package_data=True,
    description="Posture Assessment and comparison",
)
