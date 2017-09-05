#include <iostream>
#include "Python.h"
#include "main.h"
using namespace std;

static PyObject *
mainpy_run(PyObject *self, PyObject *args)
{
    const char *filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }
    run(filepath);
    return Py_None;
}

static PyMethodDef MainPyMethods[] = {
    {"run",  mainpy_run, METH_VARARGS,
     "Run on file."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initmainpy(void)
{
    PyObject *m;

    m = Py_InitModule("mainpy", MainPyMethods);
    if (m == NULL)
        return;
}


