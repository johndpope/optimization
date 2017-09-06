/*
 * RasterSpaceLib provide two methods:
 *
 * void addLines(uint** space, list<line_param> lines, int SpaceSize)
 * [[I]],[O],i
 *
 * Point2f calc_CC_Vanp(uint** space, int SpaceSize, float Normalization, int height, int width, int SubPixelRadius, int searchRange, int margin, int vp)
 * [[i]],i,f,i,i,i,i,i,i
 */

#include <iostream>
#include "Python.h"
#include "RasterSpace.hpp"
#include <list>

using namespace std;

double get_float_attr(PyObject * pyobject, const char in_char[]){
    // cout << "TEST " << PyString_AsString(PyObject_Repr(PyObject_Type(pyobject))) << endl;
    PyObject * x = PyObject_GetAttrString(pyobject, "a");
    if (PyFloat_Check(x)){
        float z = (float) PyFloat_AsDouble(x);
        // cout << "TEST " << z << endl;
        return z;
    }else{
        return -1;
    }
}

uint ** get_matrix_from_lists(PyObject * arr_object){
    int len = (int) PyList_Size(arr_object);
    uint ** c_array = (uint**)malloc(len*sizeof(uint*));
    for(int i=0; i < len; i++){
        PyObject *sublist = PyList_GetItem(arr_object, i);
        int sub_len = PyList_Size(sublist);
        c_array[i] = (uint*)malloc(sub_len*sizeof(uint));
        for(int j=0; j < sub_len; j++){
            c_array[i][j] = 0;
            c_array[i][j] = (uint) PyInt_AsLong(PyList_GetItem(sublist, j));
        }
    }
    return c_array;
}

list<line_param> get_lines(PyObject * lines_py){
    list<line_param> result_lines;
    int len = (int) PyList_Size(lines_py);
    for(int i = 0; i < len; i++){
        PyObject * pyspace = PyList_GetItem(lines_py, i);
        float a = get_float_attr(pyspace, "a");
        float b = get_float_attr(pyspace, "b");
        float c = get_float_attr(pyspace, "c");
        float d = get_float_attr(pyspace, "d");

        line_param one_space = line_param(a,b,c,d);
        result_lines.push_back(one_space);
    }
    return result_lines;
}

static PyObject *
RasterSpaceLib_addLines(PyObject *self, PyObject *args)
{
    PyObject * lines_py;
    PyObject * space_py;

    int SpaceSize;

    if (!PyArg_ParseTuple(args, "OOi", &space_py, &lines_py, &SpaceSize)) {
        return NULL;
    }

    list<line_param> lines = get_lines(lines_py);
    uint ** space = get_matrix_from_lists(space_py);
    addLines(space, lines, SpaceSize);
    return Py_None;
}

static PyObject *
RasterSpaceLib_calc_CC_Vanp(PyObject *self, PyObject *args)
{
    PyObject * space_py;
    int SpaceSize;
    float Normalization;
    int height;
    int width;
    int SubPixelRadius;
    int searchRange;
    int margin;
    int vp;

    if (!PyArg_ParseTuple(args, "Oifiiiiii", &space_py, &SpaceSize, &Normalization, &height, &width, &SubPixelRadius, &searchRange, &margin, &vp)) {
        return NULL;
    }
    uint ** space = get_matrix_from_lists(space_py);
    Point2f point = calc_CC_Vanp(space, SpaceSize, Normalization, height, width, SubPixelRadius, searchRange, margin, vp);
    return Py_BuildValue("(ii)", point.x, point.y);
}

static PyMethodDef RasterSpaceLibMethods[] = {
    {"addLines",  RasterSpaceLib_addLines, METH_VARARGS,
     "addLines"},
    {"calc_CC_Vanp",  RasterSpaceLib_calc_CC_Vanp, METH_VARARGS,
     "calc_CC_Vanp"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initRasterSpaceLib(void)
{
    PyObject *m;

    m = Py_InitModule("RasterSpaceLib", RasterSpaceLibMethods);
    if (m == NULL)
        return;
}
