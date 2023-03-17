using System;
using System.Linq;
using NConstrictor;
using NTensorFlow;

namespace NTensorFlowSample
{
    class MnistCnn
    {
        public static void Run()
        {
            float learning_rate = 0.01f;
            int epochs = 1500;
            int batch_size = 50;
            float dropout = 0.5f;

            var mnist = new TF.Keras.Datasets.MNIST();
            PyArray<double> TrainImages = (PyArray<double>)(mnist.xTrain.Reshape(60000, 784) / 255d);// float32/float32以外の演算の戻り値はfloat64になる
            PyArray<double> TestImages = (PyArray<double>)(mnist.xTest.Reshape(10000, 784) / 255d);

            Func<PyArray<byte>, PyArray<float>> ToCategorical = TF.Keras.Utils.ToCategorical;
            var TrainLabels = ToCategorical(mnist.yTrain);
            var TestLabels = ToCategorical(mnist.yTest);

            // 入力層
            var x = new PlaceHolder<float>(Py.None, 784);

            // 画像データ[n x 784]を[n x 28 x 28 x 1]に変換
            var outImg = TF.Reshape(x, new[] { -1, 28, 28, 1 });

            // 入力
            var f1 = new Variable<float>(TF.TruncatedNormal<float>(new[] { 3, 3, 1, 32 }, stddev: 0.1f));

            // 畳み込み (n, 28, 28, 32)
            var conv1 = TF.NN.Conv2d(outImg, f1, strides: new PyList { 1, 1, 1, 1 }, padding: "SAME");
            var b1 = new Variable<float>(TF.Constant(0.1f, shape: new[] { 32 }));
            var out_conv1 = TF.NN.ReLU(conv1 + b1);
            var out_pool1 = TF.NN.MaxPool(out_conv1, kSize: new PyList { 1, 2, 2, 1 }, strides: new PyList { 1, 2, 2, 1 }, padding: "SAME");

            //畳み込み (n, 28, 28, 32)
            var f2 = new Variable<float>(TF.TruncatedNormal<float>(new[] { 3, 3, 32, 64 }, stddev: 0.1f));
            var conv2 = TF.NN.Conv2d(out_pool1, f2, strides: new PyList { 1, 1, 1, 1 }, padding: "SAME");
            var b2 = new Variable<float>(TF.Constant(0.1, shape: new[] { 64 }));
            var out_conv2 = TF.NN.ReLU(conv2 + b2);
            var out_pool2 = TF.NN.MaxPool(out_conv2, kSize: new PyList { 1, 2, 2, 1 }, strides: new PyList { 1, 2, 2, 1 }, padding: "SAME");

            // ドロップアウト
            var keep_prob = new PlaceHolder<float>(Py.None);
            var out_drop = TF.NN.Dropout(out_pool2, 1 - keep_prob);

            // 全結合層
            var out_flat = TF.Reshape(out_drop, new[] { -1, 7 * 7 * 64 });
            var w_comb1 = new Variable<float>(TF.TruncatedNormal<float>(new[] { 7 * 7 * 64, 1024 }, stddev: 0.1f));
            var b_comb1 = new Variable<float>(TF.Constant(0.1, shape: new[] { 1024 }));
            var out_comb1 = TF.NN.ReLU(TF.Matmul(out_flat, w_comb1) + b_comb1);

            // 出力層
            var w_comb2 = new Variable<float>(TF.TruncatedNormal<float>(new[] { 1024, 10 }, stddev: 0.1f));
            var b_comb2 = new Variable<float>(TF.Constant(0.1, shape: new[] { 10 }));
            var outData = TF.NN.Softmax(TF.Matmul(out_comb1, w_comb2) + b_comb2);

            // 教師信号
            var t = new PlaceHolder<float>(Py.None, 10);
            
            // 誤差関数
            var b = t * TF.Log(outData);
            var a = -TF.ReduceSum(b, axis: new PyList { 1 });
            var loss = TF.ReduceMean(a);

            // ミニマイザー
            var trainOp = new TF.Train.GradientDescentOptimizer(learning_rate).Minimize(loss);

            // 評価
            var correct = TF.Equal(TF.ArgMax(outData, 1), TF.ArgMax(t, 1));
            var accuracy = TF.ReduceMean(TF.Cast<float>(correct));

            // 変数を初期化するノード
            var sess = new Session();

            foreach (int i in Enumerable.Range(0, epochs))
            {
                //オフセット=回数×バッチサイズを訓練データの行数で割った余り
                var offset = (i * batch_size) % 60000;

                //オフセットの位置からミニバッチを抽出
                PyArray<float> batch_train_images = TrainImages[offset..(offset + batch_size), ..];
                PyArray<float> batch_train_labels = TrainLabels[offset..(offset + batch_size), ..];

                PyList preResult = sess.Run(new[] { trainOp, loss }, new PyDict { { x, batch_train_images }, { t, batch_train_labels }, { keep_prob, dropout } });
                if (i % 10 == 0) Python.SimplePrint("Step:", i, "Current loss:", preResult[1]);//結果はlossの値を出力

                //100回ごとにテストデータを使用して精度を出力
                if ((i + 1) % 100 == 0)
                {
                    var acc_val = sess.Run(accuracy, new PyDict { { x, TestImages }, { t, TestLabels }, { keep_prob, 0.1 } });
                    Python.SimplePrint("Step ", i + 1, ": accuracy = ", acc_val);//結果はlossの値を出力
                }
            }
        }
    }
}
