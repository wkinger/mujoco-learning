from setuptools import setup, find_packages

setup(
    name="mujoco_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "mujoco",
    ],
)