# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="keras_utility",
    version="1.0.0",
    description='',
    packages=find_packages(),
    install_requires=['keras==2.2.4'],
    extras_require={
        "develop": ["autopep8", "pep8"]
    },
)
