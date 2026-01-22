from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(filepath:str)->List[str]:
    """
    Fetches the required libraries from a requirements file.
    :param filepath
    :return: List of requirements as strs
    """
    requirements = []
    with open(filepath, 'r') as f:
        requirements = f.readlines()
        requirements = [reqs.replace("\n","") for reqs in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.1.0',
    author='Jesmi George',
    author_email='georgejesmi03@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)