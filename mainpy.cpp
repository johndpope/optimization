#include <iostream>
#include "Python.h"
// 1. HERE I Include main.h (lazy import) your main.cpp, by headers file
#include "main.h"
using namespace std;

// 5. Main function which get params from python and run your code
static PyObject *
mainpy_run(PyObject *self, PyObject *args)
{
    const char *filepath;

    // 6. Parse arguments from python (when someone run method mainpy.run('226.mp4'))
    // we get it here, or return NULL (it says to python, that method require arguments.
    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }
    // 7. here I run method from main.cpp / run
    run(filepath);
    // 8. returns to python None
    return Py_None;
}

// 4. Here is method returns list of function
// in our case it says to python that:
// mainpy has method 'run'
// which should be reference to C++ function
// "mainpy_run" (which is defined also here)
static PyMethodDef MainPyMethods[] = {
    {"run",  mainpy_run, METH_VARARGS,
     "Run on file."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// 2. This methods run when you type import mainpy
PyMODINIT_FUNC
initmainpy(void)
{
    PyObject *m;

    // 3. Here we say to Python how method should be named, and which method returns list of functions
    m = Py_InitModule("mainpy", MainPyMethods);
    if (m == NULL)
        return;
}


