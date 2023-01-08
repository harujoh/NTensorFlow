using NConstrictor;
using System;
using System.Collections.Generic;
using System.Text;

namespace NTensorFlow
{
    public class InteractiveSession : IDisposable
    {
        private PyObject _session;

        public InteractiveSession()
        {
            _session = TF.InteractiveSession();
        }

        public void Dispose()
        {
            _session["close"].Call();
            Py.Clear(_session);
        }
    }
}
