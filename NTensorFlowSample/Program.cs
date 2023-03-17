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
            //TF.DebugInitialize(true);

            Xor.Run();
            //MnistCnn.Run();

            //何かエラーがあれば表示する
            PyErr.Print();

            Console.WriteLine("Done.");
            Console.Read();

        }
    }
}