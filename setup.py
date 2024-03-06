from setuptools import setup, find_packages

with open('requirements.txt', encoding='UTF-8') as f:
    requirements = f.read().splitlines()


setup(
    name = 'RubiksCube',
    packages = find_packages(),
    install_requires = requirements,
)