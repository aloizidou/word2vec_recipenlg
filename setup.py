from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A clean NumPy implementation of the Word2Vec skip-gram model with negative sampling trained on the RecipeNLG dataset.',
    author='Andrea Loizidou',
    license='MIT',
)
