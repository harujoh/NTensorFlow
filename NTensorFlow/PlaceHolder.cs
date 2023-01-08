using NConstrictor;
using System;
using System.Runtime.CompilerServices;

namespace NTensorFlow
{
    //https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/placeholder
    public struct PlaceHolder<T> : IDisposable
    {
        private PyObject _rawData;

        public PlaceHolder(params PyObject[] shape):this((PyList)shape) { }

        public PlaceHolder(PyList list)
        {
            _rawData = Py.None;

            PyObject type;

            if (typeof(T) == typeof(float))
            {
                type = TF.Float32;
            }
            else if (typeof(T) == typeof(int))
            {
                type = TF.Int32;
            }
            else
            {
                throw new NotImplementedException();
            }

            if(list.Count != 0)
            {
                if(list[0] == Py.None)
                {
                    _rawData = TF.PlaceHolder.Call(type);
                }
                else
                {
                    _rawData = TF.PlaceHolder.Call(type, list);
                }
            }
            else
            {
                _rawData = TF.PlaceHolder.Call(type);
            }
        }

        public static PyObject operator +(PlaceHolder<T> x, PlaceHolder<T> y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(PlaceHolder<T> x, PlaceHolder<T> y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(PlaceHolder<T> x, PlaceHolder<T> y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(PlaceHolder<T> x, PlaceHolder<T> y)
        {
            return PyNumber.TrueDivide(x, y);
        }


        //
        public static PyObject operator +(PyObject x, PlaceHolder<T> y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(PyObject x, PlaceHolder<T> y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(PyObject x, PlaceHolder<T> y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(PyObject x, PlaceHolder<T> y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        //
        public static PyObject operator +(PlaceHolder<T> x, PyObject y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(PlaceHolder<T> x, PyObject y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(PlaceHolder<T> x, PyObject y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(PlaceHolder<T> x, PyObject y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        public static implicit operator PyObject(PlaceHolder<T> variable)
        {
            return Unsafe.As<PlaceHolder<T>, PyObject>(ref variable);
        }

        public static implicit operator PlaceHolder<T>(PyObject pyObject)
        {
            return Unsafe.As<PyObject, PlaceHolder<T>>(ref pyObject);
        }

        public void Dispose()
        {
            Py.Clear(_rawData);
        }
    }
}
