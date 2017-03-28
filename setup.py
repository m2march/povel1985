from setuptools import setup, find_packages

setup(
    name='povel1985',
    description='Library package Povel 1985 clock induction model',
    packages=find_packages(),
    namespace_packages=['m2'],
    install_requires=[],
    dependency_links=['file://../tht']
)
