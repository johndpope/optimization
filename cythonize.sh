#!/usr/bin/env bash
cd build
cython -a ../RasterSpace.pyx -o RasterSpace.c
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/include/python2.7 -o RasterSpace.so RasterSpace.c
cp RasterSpace.so ../
