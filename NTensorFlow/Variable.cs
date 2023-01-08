using System;
using NConstrictor;
using System.Runtime.CompilerServices;

namespace NTensorFlow
{
    public struct Variable<T> : IDisposable
    {
        private PyObject _rawData;

        //public PyArray<T> Grad
        //{
        //    get { return _rawData["grad"]; }
        //    set { _rawData["grad"] = value; }
        //}

        //public PyArray<T> Data
        //{
        //    get { return _rawData["data"]; }
        //    set { _rawData["data"] = value; }
        //}

        //public PyArray<T> Shape
        //{
        //    get { return _rawData["shape"]; }
        //}

        public Variable(params PyObject[] args)
        {
            _rawData = TF.Variable.Call(args);
        }

        public Variable(PyDict kw)
        {
            _rawData = TF.Variable.Call(kw);
        }

        public Variable(PyObject args, PyDict kw)
        {
            _rawData = TF.Variable.Call(args, kw);
        }

        public Variable(PyObject[] args, PyDict kw)
        {
            _rawData = TF.Variable.Call(args, kw);
        }

        //public void Backward()
        //{
        //    _rawData["backward"].Call();
        //}

        public static PyObject operator +(Variable<T> x, Variable<T> y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(Variable<T> x, Variable<T> y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(Variable<T> x, Variable<T> y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(Variable<T> x, Variable<T> y)
        {
            return PyNumber.TrueDivide(x, y);
        }


        //
        public static PyObject operator +(PyObject x, Variable<T> y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(PyObject x, Variable<T> y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(PyObject x, Variable<T> y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(PyObject x, Variable<T> y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        //
        public static PyObject operator +(Variable<T> x, PyObject y)
        {
            return PyNumber.Add(x, y);
        }

        public static PyObject operator -(Variable<T> x, PyObject y)
        {
            return PyNumber.Subtract(x, y);
        }

        public static PyObject operator *(Variable<T> x, PyObject y)
        {
            return PyNumber.Multiply(x, y);
        }

        public static PyObject operator /(Variable<T> x, PyObject y)
        {
            return PyNumber.TrueDivide(x, y);
        }

        public static implicit operator PyObject(Variable<T> variable)
        {
            return Unsafe.As<Variable<T>, PyObject>(ref variable);
        }

        public static implicit operator Variable<T>(PyObject pyObject)
        {
            return Unsafe.As<PyObject, Variable<T>>(ref pyObject);
        }

        public void Dispose()
        {
            Py.Clear(_rawData);
        }
    }
}
