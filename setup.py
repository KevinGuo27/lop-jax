from setuptools import setup, find_packages

setup(
    name="lop-jax",
    version="0.0.1",
    packages=find_packages(),   # finds permuted_mnist, utils, etc.
    install_requires=[
        "jax",
        "distrax",
        "gymnax",
        "typed-argument-parser",
        "torch",
        "torchvision",
        "tqdm",
        "optax",
        "numpy==1.26.4",
        "matplotlib",
    ],
)