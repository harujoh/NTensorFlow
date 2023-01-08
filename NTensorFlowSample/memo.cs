//PyObject truncatedNormal = TF.TruncatedNormal<float>(new[] { 2, HIDDEN_NODES }, stddev: 1.0f / MathF.Sqrt(2.0f));

//var session = new InteractiveSession();
//var val = truncatedNormal["eval"].Call();
//Python.Print(val);


//var init = TF.InitializeAllVariables();
//using (var session = new Session())
//{
//    session.Run(init);
//    var a = session.Run(wHidden);
//    var b = session.Run(bHidden);
//    Python.Print(a);
//    Python.Print(b);
//}


//Console.WriteLine(TF.Version);


//PyObject truncatedNormal = TF.TruncatedNormal(new[] { 2, HIDDEN_NODES }, stddev: 1.0f / MathF.Sqrt(2.0f));
//
//var session = new InteractiveSession();
//var val = truncatedNormal["eval"].Call();
//Python.Print(val);



//var a = new PlaceHolder<int>(2);
//var b = new PlaceHolder<int>(2);

//var c = a * b;

//PyArray<int> valA = new[] { 2, 1 };
//PyArray<int> valB = new[] { 3, 4 };

//var dic = new Dictionary<PyObject, PyObject>
//{
//    { a, valA },
//    { b, valB }
//};

//var session = new Session();
//var result = session.Run(c, dic);

//Python.Print(result);



//var session = new InteractiveSession();
//PyArray<float> list = new [,] { { 1.0f, 1.0f }, { 2.0f, 2.0f } };
//var x = TF.Constant(list);
//var val = TF.ReduceMean(x);
//Python.Print(val["eval"].Call());


/*
    logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
    labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    [0.16984604, 0.82474494]
 */
//var session = new InteractiveSession();
//PyList logits = new PyList(new PyObject[,] { { 4.0f, 2.0f, 1.0f }, { 0.0f, 5.0f, 1.0f } });
//PyList labels = new PyList(new PyObject[,] { { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.8f, 0.2f } });
//var val = TF.NN.SoftmaxCrossEntropyWithLogits(labels, logits);
//Python.Print(val);
//Python.Print(val["eval"].Call());
