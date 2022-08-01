import os
from setuptools import setup, find_packages


# Helper function to load up readme as long description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
        name = 'neurosymPI',
        version = '0.1',
        author='Nicole Dumont',
        author_email='ns2dumont@uwaterloo.ca',
        description=('Code for "A model of path integration that connects neural and symbolic representation", presented at CogSci2022'),
        license = 'TBD',
        keywords = '',
        url='http://github.com/nsdumont/neurosymPI',
        packages=find_packages(),
        long_description=read('README.md'),
        install_requires=[
            'numpy', 'scipy',
            'matplotlib', 'nengo', 'pytry'
            ]
)
