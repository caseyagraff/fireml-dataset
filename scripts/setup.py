from setuptools import setup

from utils import parse_toml

project = parse_toml()

project_name = project["name"].strip('"')
project_version = project["version"].strip('"')

setup(
    name=project_name, version=project_version, packages=[project_name],
)
