using System;
using NConstrictor;
using NTensorFlow;

namespace NTensorFlowSample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Python.Initialize(true);
            TF.Initialize(true);

            Xor.Run();
            MnistCnn.Run();

            Console.WriteLine("Done.");
            Console.Read();
        }
    }
}