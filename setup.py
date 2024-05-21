from setuptools import find_packages, setup

import os

# 获取当前目录
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


install_requires = ['diffusers-interpret @ file://{}'.format(os.path.join(CURRENT_DIR, "diffusers_interpret")),
    "detectron2 @ git+https://github.com/baaivision/EVA.git#subdirectory=EVA-02/det"]

setup(
    name='nam',
    version='0.1.0',
    packages=find_packages(),
    description='Neuron Attribution method tailored for MLLMs',
    author='Ananymous',
    license='MIT',
    python_requires=">=3.8",
    install_requires=install_requires
)
