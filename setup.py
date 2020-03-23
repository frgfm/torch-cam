#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Package installation setup
"""

import os
import subprocess
from setuptools import setup, find_packages


version = '0.1.0a0'
sha = 'Unknown'
package_name = 'torchcam'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'torchcam', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()

with open('README.md') as f:
    readme = f.read()

requirements = [
    'torch>=1.1.0',
    'numpy>=1.14.0',
    'pillow>=5.0.0',
    'matplotlib>=3.0.0'
]

setup(
    # Metadata
    name=package_name,
    version=version,
    author='François-Guillaume Fernandez',
    description='Class activation maps for your PyTorch CNN models',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/frgfm/torch-cam',
    download_url='https://github.com/frgfm/torch-cam/tags',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=['pytorch', 'deep learning', 'cnn', 'convolution', 'activation', 'gradcam'],

    # Package info
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=requirements,
    package_data={'': ['LICENSE']}
)
