#include <Python.h>

static PyObject *module_func(PyObject *self, PyObject *args) {
   int i;
   double d;
   char *s;

   if (!PyArg_ParseTuple(args, "ids", &i, &d, &s)) {
      return NULL;
   }
   
   /* Do something interesting here. */
   Py_RETURN_NONE;
}
