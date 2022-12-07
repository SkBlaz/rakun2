""" Main setup file """

from os import path
from setuptools import setup, find_packages


def parse_requirements(file):
    """ Generic requirement parser """

    required_packages = []
    with open(path.join(path.dirname(__file__), file), encoding="utf-8") as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


setup(name='rakun2',
      version='0.25',
      description=
      "RaKUn 2.0; Better faster stronger lighter",
      url='http://github.com/skblaz/rakun2',
      author='Blaž Škrlj',
      author_email='blaz.skrlj@ijs.si',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=parse_requirements("requirements.txt"),
      include_package_data=True)
