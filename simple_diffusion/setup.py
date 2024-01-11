from setuptools import setup, find_packages
import os

setup(
    name="simple-diffusion",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    tests_require=[
        'unittest'
    ],
    #test_suite="tests.test_slurm_manager",  # Point to the specific test module
    author="Lukas",
    author_email="lherron@umd.edu",
    description="A Python module to study molecular dynamics simulations using Thermodynamic Maps",
    #long_description=open('README.md').read(),
    #long_description_content_type="text/markdown",
    #url="https://github.com/yourusername/slurmmanager",  # Update with your repository URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

