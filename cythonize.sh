#!/usr/bin/env bash
cd build
cython -a ../RasterSpace.pyx -o RasterSpace.c
gcc -stdlib=libc++ -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -o RasterSpace.so RasterSpace.c -v
cp RasterSpace.so ../
