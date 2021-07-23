from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np

from utils import parse_toml

project = parse_toml()

project_name = project["name"].strip('"')

ext_modules = [
    Extension(
        "*",
        [
            f"{project_name}/processing/**/*.pyx",
        ],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]


# num_workers?

setup(
    name=project_name,
    # ext_modules=cythonize(f"{project_name}/processing/**/*.pyx"),
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
)
