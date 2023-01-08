using System;
using System.Linq;
using NConstrictor;
using NTensorFlow;

namespace NTensorFlowSample
{
    internal class Xor
    {
        public static void Run()
        {

            const int HIDDEN_NODES = 10;

            var x = new PlaceHolder<float>(Py.None, 2);
            var wHidden = new Variable<float>(TF.TruncatedNormal<float>(new[] { 2, HIDDEN_NODES }, stddev: 1.0f / MathF.Sqrt(2.0f)));
            var bHidden = new Variable<float>(TF.Zeros<float>(HIDDEN_NODES));
            var hidden = TF.NN.ReLU(TF.Matmul(x, wHidden) + bHidden);

            var wLogits = new Variable<float>(TF.TruncatedNormal<float>(new[] { HIDDEN_NODES, 2 }, stddev: 1.0f / MathF.Sqrt(HIDDEN_NODES)));
            var bLogits = new Variable<float>(TF.Zeros<float>(2));
            var logits = TF.Matmul(hidden, wLogits) + bLogits;

            var y = TF.NN.Softmax(logits);

            var yInput = new PlaceHolder<float>(Py.None, 2);

            var crossEntropy = TF.NN.SoftmaxCrossEntropyWithLogits(yInput, logits);
            var loss = TF.ReduceMean(crossEntropy);

            var optimizer = new TF.Train.GradientDescentOptimizer(0.2);
            var trainOp = optimizer.Minimize(loss);

            var sess = new Session();

            PyArray<float> xTrain = new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            PyArray<float> yTrain = new float[,] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } };

            foreach (var i in Enumerable.Range(0, 500))
            {
                PyList preResult = sess.Run(new[] { trainOp, loss }, new PyDict { { x, xTrain }, { yInput, yTrain } });//trainOpの出力はいらないが実行が必要

                if (i % 10 == 0)
                {
                    Python.Print("Step:", i, "Current loss:", preResult[1]);//結果はlossの値を出力

                    foreach (PyList xInput in new [] { new[] { 0.0f, 0.0f }, new[] { 0.0f, 1.0f }, new[] { 1.0f, 0.0f }, new[] { 1.0f, 1.0f }})
                    {
                        var gy = sess.Run(y, new PyDict { { x, new PyList{ xInput } } });//受け取り側は2次元なので次元を拡張
                        Python.SimplePrint(xInput, gy);
                    }
                }
            }
        }
    }
}
