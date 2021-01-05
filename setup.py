from setuptools import setup, Extension, find_packages
import numpy as np
from Cython.Build import cythonize


setup(
    name="bmf_tool",
    version="1.0.0",
    description="Bipartite-Motif-Finder to find co-occuring over-represented patterns in RNA sequences",
    license="GPLv3",
    author="Salma Sohrabi-Jahromi",
    author_email="ssohrab@mpibpc.mpg.de",
    packages=find_packages(),
    install_requires=['numpy', 'cython', 'scipy', 'pandas', 'biopython', 'sklearn','matplotlib', 'seaborn'],
    python_requires='>3.6',
    ext_modules=cythonize([
        Extension('bmf_tool.utils.dp_z',
              sources=['bmf_tool/utils/dp_z.pyx'],
              extra_compile_args=['-std=c99', '-mavx2', '-mfpmath=sse', '-DAVX2=1'],
              include_dirs=[np.get_include()]
              )
    ]),
    entry_points={
        'console_scripts': [
            'bmf=bmf_tool.bipartite_finder:main',
            'bmf_logo=bmf_tool.utils.plot_logo:main'
        ]
    }
)