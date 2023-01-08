using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NConstrictor;
using NTensorFlow;

namespace NTensorFlowSample
{
    internal class ResNet
    {
        public static void Run()
        {
            Python.Initialize(true);
            TF.Initialize(true);

        }
    }
}
