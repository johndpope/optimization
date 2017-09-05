#!/usr/bin/env bash
c++ mainpy.cpp main.cpp -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_video -o mainpy.so
