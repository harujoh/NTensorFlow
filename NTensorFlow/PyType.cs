using NConstrictor;
using System.Runtime.CompilerServices;

namespace NTensorFlow
{
    public struct PyType
    {
        private PyObject _rawData;

        public static implicit operator PyObject(PyType variable)
        {
            return Unsafe.As<PyType, PyObject>(ref variable);
        }

        public static implicit operator PyType(PyObject pyObject)
        {
            return Unsafe.As<PyObject, PyType>(ref pyObject);
        }

        public static implicit operator PyType(PyDynamic pyObject)
        {
            return Unsafe.As<PyObject, PyType>(ref pyObject._pyObject);
        }

    }
}
