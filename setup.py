# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from setuptools import setup, find_packages

from script import __version__

setup(
    name='modbank',
    version=__version__,

    url='https://github.com/zhu00121/modulation_filterbanks/tree/main',
    author='Yi Zhu',
    author_email='Yi.Zhu@inrs.ca',

    packages=find_packages(),
)