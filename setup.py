from distutils.core import setup, Extension

setup(name="shift_function", version="0.0",
	ext_modules = [Extension("example", ["shiftfunc.c"])])
#setup(name="mandelbrot", version="0.0",
#	ext_modules = [Extension("mandelbrot", ["mandelbrot.c"])])
