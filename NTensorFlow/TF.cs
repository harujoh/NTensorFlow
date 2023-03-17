using NConstrictor;
using System;
using System.IO;

namespace NTensorFlow
{
    //https://www.tensorflow.org/versions/r1.15/api_docs/python/tf
    public static class TF
    {
        private static dynamic _tf;
        public static string Version;

        public static PyObject PlaceHolder;

        internal static PyObject Variable;

        public static PyType Float32;
        public static PyType Int32;

        public class Keras
        {
            public class Utils
            {
                //タイプ指定がない場合はfloatがデフォルト
                public static PyArray<float> ToCategorical<T>(PyArray<T> pyArray)
                {
                    return (PyArray<float>)_tf.keras.utils.to_categorical(pyArray);
                }

                //タイプ指定はジェネリックで行う
                public static PyArray<Tout> ToCategorical<Tin, Tout>(PyArray<Tin> pyArray)
                {
                    return (PyArray<Tout>)_tf.keras.utils.to_categorical(pyArray);
                }
            }

            public class Datasets
            {
                public class CIFAIR10
                {
                    private dynamic _cifair10;

                    public PyArray<byte> xTrain;
                    public PyArray<byte> yTrain;
                    public PyArray<byte> xTest;
                    public PyArray<byte> yTest;

                    public CIFAIR10()
                    {
                        _cifair10 = _tf.keras.datasets.cifair10;
                        PyTuple pyTuple = _cifair10.load_data();
                        PyObject[] preTuple = pyTuple.UnPack();
                        PyTuple[] result = { preTuple[0], preTuple[1] };

                        xTrain = (PyArray<byte>)result[0][0];
                        yTrain = (PyArray<byte>)result[0][1];
                        xTest = (PyArray<byte>)result[1][0];
                        yTest = (PyArray<byte>)result[1][1];
                    }
                }

                public class MNIST
                {
                    private dynamic _mnist;

                    public PyArray<byte> xTrain;
                    public PyArray<byte> yTrain;
                    public PyArray<byte> xTest;
                    public PyArray<byte> yTest;

                    public MNIST()
                    {
                        _mnist = _tf.keras.datasets.mnist;
                        PyTuple pyTuple = _mnist.load_data();
                        PyObject[] preTuple = pyTuple.UnPack();
                        PyTuple[] result = { preTuple[0], preTuple[1] };

                        xTrain = (PyArray<byte>)result[0][0];
                        yTrain = (PyArray<byte>)result[0][1];
                        xTest = (PyArray<byte>)result[1][0];
                        yTest = (PyArray<byte>)result[1][1];
                    }
                }
            }
        }

        public class NN
        {
            public static PyObject MaxPool(PyObject input, PyObject kSize, PyObject strides, PyObject padding, PyObject dataFormat = default, PyObject name = default)
            {
                PyDict feed_dict = new PyDict();
                if (dataFormat != default) feed_dict["data_format"] = dataFormat;
                if (name != default) feed_dict["name"] = name;

                return _tf.nn.max_pool(input, kSize, strides, padding, feed_dict);
            }

            public static PyObject AvgPool(PyObject input, PyObject kSize, PyObject strides, PyObject padding, PyObject dataFormat = default, PyObject name = default)
            {
                PyDict feed_dict = new PyDict();
                if (dataFormat != default) feed_dict["data_format"] = dataFormat;
                if (name != default) feed_dict["name"] = name;

                return _tf.nn.avg_pool(input, kSize, strides, padding, feed_dict);
            }

            public static PyObject Conv2d(PyObject input, PyObject filters, PyObject strides, PyObject padding, PyObject dataFormat = default, PyObject dilations = default, PyObject name = default)
            {
                PyDict feed_dict = new PyDict();

                if (dataFormat != default) feed_dict["data_format"] = dataFormat;
                if (dilations != default) feed_dict["dilations"] = dilations;
                if (name != default) feed_dict["name"] = name;

                return _tf.nn.conv2d(input, filters, strides, padding, feed_dict);
            }

            public static PyObject Moments(PyObject x, PyObject axes, PyObject shift = default, PyObject keepDims = default, PyObject name = default)
            {
                PyDict feed_dict = new PyDict();

                if (shift != default) feed_dict["shift"] = shift;
                if (keepDims != default) feed_dict["keepdims"] = keepDims;
                if (name != default) feed_dict["name"] = name;

                return _tf.nn.moments(x, axes, feed_dict);
            }

            public static PyObject BatchNormalization(PyObject x, PyObject mean, PyObject variance, PyObject offset, PyObject scale, PyObject variance_epsilon, PyObject name = default)
            {
                PyDict feed_dict = new PyDict();
                if (name != default) feed_dict["name"] = name;

                return _tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, feed_dict);
            }

            public static PyObject Dropout(PyObject x, PyObject rate)
            {
                return _tf.nn.dropout(x, rate);
            }

            public static PyObject ReLU(PyObject features)
            {
                return _tf.nn.relu(features);
            }

            //https://www.tensorflow.org/api_docs/python/tf/nn/softmax
            public static PyObject Softmax(PyObject logits)
            {
                return _tf.nn.softmax(logits);
            }

            //https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            //Note that to avoid confusion, it is required to pass only named arguments to this function.
            //訳:混乱を避けるため、この関数には名前付きで引数を渡す必要がある
            public static PyObject SoftmaxCrossEntropyWithLogits(PyObject labels, PyObject logits)
            {
                PyDict feed_dict = new PyDict
                {
                    {"labels", labels},
                    {"logits", logits}
                };

                return _tf.nn.softmax_cross_entropy_with_logits(feed_dict);
            }

            //https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
            //こちらは名前付きが強制ではなく推奨
            public static PyObject SparseSoftmaxCrossEntropyWithLogits(PyObject labels, PyObject logits)
            {
                PyDict feed_dict = new PyDict
                {
                    {"labels", labels},
                    {"logits", logits}
                };

                return _tf.nn.softmax_cross_entropy_with_logits(feed_dict);
            }
        }

        public class Train
        {
            public abstract class Optimizer
            {
                protected dynamic _optimizer;

                public PyObject Minimize(PyObject loss)
                {
                    return _optimizer.minimize(loss);
                }
            }

            public class GradientDescentOptimizer : Optimizer
            {
                public GradientDescentOptimizer(PyObject learningRate)
                {
                    _optimizer = (PyDynamic)_tf.train.GradientDescentOptimizer(learningRate);
                }
            }

            public static class SavedModel
            {
                //https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/load
                public static void Load(Session sess, PyDict tags, string export_dir)
                {
                    _tf.saved_model.load((PyObject)sess, tags, export_dir);
                }

                //https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/simple_save
                public static void SimpleSave(PyObject sess, PyObject export_dir, PyObject[] inputs, PyObject[] outputs)
                {
                    PyDict inputDict = new PyDict();
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        inputDict[Python.GetName(inputs[i])] = inputs[i];
                    }

                    PyDict outputDict = new PyDict();
                    for (int i = 0; i < outputs.Length; i++)
                    {
                        outputDict[Python.GetName(outputs[i])] = outputs[i];
                    }

                    _tf.saved_model.simple_save(sess, export_dir, (PyObject)inputDict, (PyObject)outputDict);
                }

                public static void SimpleSave(Session sess, string export_dir, PyDict inputs, PyDict outputs)
                {
                    if (Directory.Exists(export_dir))
                    {
                        throw new Exception(export_dir + "は既に同名のフォルダが存在します");
                    }
                    _tf.saved_model.simple_save(sess, export_dir, (PyObject)inputs, (PyObject)outputs);
                }

                public static void SimpleSave(Session sess, string export_dir, PyObject[] inputs, PyObject output)
                {
                    if (Directory.Exists(export_dir))
                    {
                        throw new Exception(export_dir + "は既に同名のフォルダが存在します");
                    }
                    SimpleSave(sess,export_dir,inputs, new[] { output });
                }

                public static void SimpleSave(Session sess, string export_dir, PyObject input, PyObject[] outputs)
                {
                    if (Directory.Exists(export_dir))
                    {
                        throw new Exception(export_dir + "は既に同名のフォルダが存在します");
                    }
                    SimpleSave(sess, export_dir, new[] { input }, outputs);
                }

                public static void SimpleSave(Session sess, string export_dir, PyObject input, PyObject output)
                {
                    if(Directory.Exists(export_dir))
                    {
                        throw new Exception(export_dir + "は既に同名のフォルダが存在します");
                    }
                    SimpleSave(sess, export_dir, new[] { input }, new []{ output });
                }
            }
        }

        /*
            c = tf.group(a,b)と記述した場合以下と同義
            with tf.control_depencies([a, b])
                c = tf.no_op()
        */
        public static PyObject Group(PyObject inputs, PyDict kwArgs)
        {
            return _tf.group(inputs, kwArgs);
        }

        //入力と同じ形状と内容を持つ Tensor を返します
        public static PyObject Identity(PyObject input)
        {
            return _tf.identity(input);
        }

        //with句で使用される
        //こちらは順番が制御されるがGroupは毎回挙動が異なるらしい
        // https://qiita.com/jack_ama/items/6722df977f531124026d
        public static PyObject ControlDepencies(PyObject control_inputs)
        {
            return _tf.control_depencies(control_inputs);
        }

        public static PyObject Cast<T>(PyObject x)
        {
            return _tf.cast(x, Dtype.GetDtype(typeof(T)));
        }


        public static PyObject Matmul(PyObject a, PyObject b)
        {
            return _tf.linalg.matmul(a, b);
        }

        public static PyObject ReduceMean(PyObject inputTensor)
        {
            return _tf.math.reduce_mean(inputTensor);
        }

        public static PyObject ReduceSum(PyObject inputTensor, PyObject axis)
        {
            return _tf.math.reduce_sum(inputTensor, axis);
        }

        public static PyObject Equal(PyObject x, PyObject y)
        {
            return _tf.math.equal(x, y);
        }

        public static PyObject ArgMax(PyObject input, PyObject axis)
        {
            return _tf.math.argmax(input, axis);
        }

        public static PyObject Log(PyObject inputTensor)
        {
            return _tf.math.log(inputTensor);
        }

        //https://www.tensorflow.org/api_docs/python/tf/compat/v1/initialize_all_variables
        //非推奨 -> tf.global_variables_initializer
        //public static PyObject InitializeAllVariables()
        //{
        //    return _tf.initialize_all_variables();
        //}

        public static PyObject GlobalVariablesInitializer()
        {
            return _tf.global_variables_initializer();
        }

        public static PyObject Session()
        {
            return _tf.Session();
        }

        public static PyObject InteractiveSession()
        {
            return _tf.InteractiveSession();
        }

        //https://www.tensorflow.org/api_docs/python/tf/zeros
        public static PyObject Zeros<T>(params int[] shape)
        {
            return _tf.zeros((PyArray<int>)shape);
        }

        //https://www.tensorflow.org/api_docs/python/tf/constant
        public static PyObject Constant(PyObject value)
        {
            return _tf.constant(value);
        }

        //本来第二引数はdtype
        public static PyObject Constant(PyObject value, int[] shape)
        {
            PyDict feed_dict = new PyDict
            {
                {"shape", (PyArray<int>)shape},
            };

            return _tf.constant(value, feed_dict);
        }

        public static PyObject Reshape(PyObject value, int[] shape)
        {
            return _tf.reshape(value, (PyArray<int>)shape);
        }

        public static PyObject TruncatedNormal<T>(int[] shape, float mean = 0.0f, float stddev = 1.0f)
        {
            PyDict feed_dict = new PyDict
            {
                {"mean", mean},
                {"stddev", stddev}
            };

            return _tf.random.truncated_normal((PyArray<int>)shape, feed_dict);
        }

        public static void DebugInitialize(bool writeLog = false)
        {
            if (writeLog) Console.WriteLine("Currently warming up TensorFlow.");

            PyDynamic tf = PyImport.ImportModule("tensorflow.compat.v1");
            Python.Main["tf"] = tf;
            _tf = tf;

            //deprecationのメッセージが出力されると、そのメッセージが原因で関数が実行されないことがあるため停止
            dynamic deprecation = (PyDynamic)PyImport.ImportModule("tensorflow.python.util.deprecation");
            deprecation._PRINT_DEPRECATION_WARNINGS = false;

            _tf.disable_v2_behavior();

            Version = _tf.__version__.ToString();

            Variable = _tf.Variable;

            Float32 = _tf.float32;
            Int32 = _tf.int32;

            PlaceHolder = _tf.placeholder;

            if (writeLog)
            {
                Console.WriteLine("TensorFlow warming has been completed.");
                Console.WriteLine("TensorFlow version : " + TF.Version + Environment.NewLine);
            }
        }

        public static void Initialize(bool writeLog = false)
        {
            if (writeLog) Console.WriteLine("Currently warming up TensorFlow.");

            //エラーメッセージを止める①
            Python.Run(
@"os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)", false);
            //todo not working
            //var environ = Python.Os["environ"];
            //environ["TF_CPP_MIN_LOG_LEVEL"] = "3";
            //Dictionary<PyObject, PyObject> dic = new Dictionary<PyObject, PyObject>()
            //{
            //    {"action", "ignore"},
            //    {"category", Python.Builtins["FutureWarning"]},
            //};

            //Python.Warnings["simplefilter"].Call(dic);

            //dic["category"] = Python.Builtins["Warning"];
            //Python.Warnings["simplefilter"].Call(dic);


            //TensorFlowをインポート V1が必要なので
            //_tf = PyImport.ImportModule("tensorflow");
            PyDynamic tf = PyImport.ImportModule("tensorflow.compat.v1");
            Python.Main["tf"] = tf;
            _tf = tf;

            //エラーメッセージを止める②
            Python.Run(
@"tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)", false);
            //todo not working.
            //_tf["get_logger"].Call()["setLevel"].Call("INFO");
            //_tf["autograph"]["set_verbosity"].Call(0);
            //_tf["get_logger"].Call()["setLevel"].Call(Python.Logging["ERROR"]);

            //deprecationのメッセージが出力されると、そのメッセージが原因で関数が実行されないことがあるため停止
            dynamic deprecation = (PyDynamic)PyImport.ImportModule("tensorflow.python.util.deprecation");
            deprecation._PRINT_DEPRECATION_WARNINGS = false;

            _tf.disable_v2_behavior();

            //disable_v2_behaviorで実行されるので不要
            //_tf["disable_eager_execution"].Call();
            //_tf["disable_resource_variables"].Call();

            Version = _tf.__version__.ToString();

            Variable = _tf.Variable;

            Float32 = _tf.float32;
            Int32 = _tf.int32;

            PlaceHolder = _tf.placeholder;

            if (writeLog)
            {
                Console.WriteLine("TensorFlow warming has been completed.");
                Console.WriteLine("TensorFlow version : " + TF.Version + Environment.NewLine);
            }
        }
    }
}
