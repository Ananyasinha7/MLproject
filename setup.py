from setuptools import find_packages,setup
from typing import List

hyphen_edot='-e .'
def get_requirements(file_path:str)->List[str]:
    "this function will return list of requirements"
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if hyphen_edot in requirements:
            requirements.remove(hyphen_edot)
    return requirements
        

setup(
    name='mlproject',
    version='0.0.1',
    author='Ananya',
    author_email='anany08a@gmail.com',
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=get_requirements('requirements.txt') )
