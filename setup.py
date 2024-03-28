from setuptools import setup, find_packages

# Find requirements.txt
with open('requirements.txt', encoding='UTF-8') as f:
    requirements = f.read().splitlines()

# Setup dependencies
setup(
    name = 'RubiksCube',
    packages = find_packages(),
    install_requires = requirements,
)