from setuptools import setup, find_packages
import shutil

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

PACKAGENAME = 'vaexede'

setup(
    name='vaexede',
    version="0.1.0",
    author='Davide Piras',
    author_email='dr.davide.piras@gmail.com',
    description='Trained models for EDE and LCDM VAEs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dpiras/VAExEDE',
    license='GNU General Public License v3.0 (GPLv3)',
    packages=find_packages(),
    package_data= {'vaexede': ['trained_models/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=['tensorflow',
                      'keras<3',
                      'matplotlib',
                      'numpy']
                      )
