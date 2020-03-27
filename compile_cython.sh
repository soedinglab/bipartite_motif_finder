NP_INC="$(python -c 'import numpy;print(numpy.get_include())')"
export CFLAGS="$CFLAGS -I$NP_INC -I . -std=c99 -mavx2 -mfpmath=sse -DAVX2=1"


cythonize -f -i dp_z.pyx