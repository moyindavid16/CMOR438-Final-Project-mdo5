from setuptools import setup, find_packages

setup(
    name="my_ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest'
    ]
)