using System;
using NConstrictor;

namespace NTensorFlow
{
    //https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Session
    public class Session : IDisposable
    {
        private dynamic _session;

        //グラフを引数に取る初期化もある
        //public Session(Graph g){}

        public Session(bool initialize = true)
        {
            _session = (PyDynamic)TF.Session();
            if(initialize)
            {
                Run(TF.GlobalVariablesInitializer());//
                //Run(TF.InitializeAllVariables());//非推奨
            }
        }

        public PyList Run(PyObject fetches, PyDict feedDict)
        {
            PyDict feed_dict = new PyDict
            {
                {"feed_dict", feedDict}
            };
            
            return (PyList)_session.run(fetches, feed_dict);
        }

        public PyList Run(PyObject fetches)
        {
            return (PyList)_session.run(fetches);
        }

        public PyList Run(PyList fetches, PyDict feedDict)
        {
            PyDict feed_dict = new PyDict
            {
                {"feed_dict", feedDict}
            };

            return (PyList)_session.run(fetches, feed_dict);
        }

        public PyList Run(PyList fetches)
        {
            return (PyList)_session.run(fetches);
        }

        public void Close()
        {
            _session.close();
        }

        public void Dispose()
        {
            _session.close();
            Py.Clear(_session);
        }
    }
}
