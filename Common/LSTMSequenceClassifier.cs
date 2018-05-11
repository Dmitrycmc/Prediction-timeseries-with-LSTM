using CNTK;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO; 

namespace CNTK.CSTrainingExamples
{

    public class Points
    {
        public float[] X, Y;

        public Points(float[] _X, float[] _Y)
        {
            X = _X;
            Y = _Y;
        }
    }


    public class Set
    {
        public float[][] train, valid, test;

        public Set(float[][] _train, float[][] _valid, float[][] _test)
        {
            train = _train;
            valid = _valid;
            test = _test;
        }

    }
    
    /// <summary>
    /// This class shows how to build a recurrent neural network model from ground up and train the model. 
    /// </summary>
    public class LSTMSequence
    {
        /// <summary>
        /// Execution folder is: CNTK/x64/BuildFolder
        /// Data folder is: CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data
        /// </summary>

        DBParser db;
        int siteId;
        Func<double, double> fun;
        bool advanced_input;
        DeviceDescriptor device;

        public LSTMSequence(DBParser _db, int _siteId, DeviceDescriptor _device, bool _advanced_input = false)
        {
            db = _db;
            siteId = _siteId;
            advanced_input = _advanced_input;
            fun = (x) => db.f(siteId, "lpC", x);
            device = _device;
        }

        private static float[] asBatch(float[][] data, int start, int count)
        {
            var lst = new List<float>();
            for (int i = start; i < start + count; i++)
            {
                if (i >= data.Length)
                    break;

                lst.AddRange(data[i]);
            }
            return lst.ToArray();
        }


        private static IEnumerable<Points> nextBatch(float[][] X, float[][] Y, int mMSize)
        {



            for (int i = 0; i <= X.Length - 1; i += mMSize)
            {
                var size = X.Length - i;
                if (size > 0 && size > mMSize)
                    size = mMSize;

                var x = asBatch(X, i, size);
                var y = asBatch(Y, i, size);

                yield return new Points(x, y);
            }

        }

        public Function CreateModel(Variable input, int outDim, int LSTMDim, int cellDim, string outputName)
        {

            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            //creating LSTM cell for each input variables
            Function LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
                input,
                new int[] { LSTMDim },
                new int[] { cellDim },
                pastValueRecurrenceHook,
                pastValueRecurrenceHook).Item1;

            //after the LSTM sequence is created return the last cell in order to continue generating the network
            Function lastCell = CNTKLib.SequenceLast(LSTMFunction);

            //implement drop out for 10%
            var dropOut = CNTKLib.Dropout(lastCell, 0.2, 1);

            //create last dense layer before output
            var outputLayer = FullyConnectedLinearLayer(dropOut, outDim, outputName);

            return outputLayer;
        }

        public Function FullyConnectedLinearLayer(Variable input, int outputDim, string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            //
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1);

            int[] s = { outputDim, inputDim };

            //
            var timesParam = new Parameter((NDShape)s, DataType.Float, glorotInit, device, "timesParam");
            //
            var timesFunction = CNTKLib.Times(timesParam, input, "times");
            //
            int[] s2 = { outputDim };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }

        static Set splitData(float[][] data, float valSize = 0.1f, float testSize = 0.1f)
        {
            //calculate
            var posTest = (int)(data.Length * (1 - testSize));
            var posVal = (int)(posTest * (1 - valSize));

            return new Set(data.Skip(0).Take(posVal).ToArray(), data.Skip(posVal).Take(posTest - posVal).ToArray(), data.Skip(posTest).ToArray());
        }


        Dictionary<string, Set> loadData(int timeSteps, string featuresName, string labelsName, Func<double, double> fun)
        {
            int numx = 200; 
            float dx = 8;
            int timeShift = 1;

            ////fill data
            float[] xsin = new float[numx]; //all data
            for (int l = 0; l < numx; l++)
            {
                xsin[l] = (float)fun(dx * l);
            }

            //split data on training and testing part
            var a = new float[xsin.Length - timeShift];
            var b = new float[xsin.Length - timeShift];

            for (int l = 0; l < xsin.Length; l++)
            {
                //
                if (l < xsin.Length - timeShift)
                    a[l] = xsin[l];

                //
                if (l >= timeShift)
                    b[l - timeShift] = xsin[l];
            }

            //make arrays of data
            var a1 = new List<float[]>();
            var b1 = new List<float[]>();
            for (int i = 0; i < a.Length - timeSteps + 1; i++)
            {
                //features
                float[] row;
                if (advanced_input)
                {
                    row = new float[timeSteps + 2];
                    row[0] = (float)db.f(siteId, "MLD", dx * (i + timeSteps));
                    row[1] = (float)db.f(siteId, "light", dx * (i + timeSteps));
                    for (int j = 0; j < timeSteps; j++)
                        row[j + 2] = a[i + j];
                } else
                {
                    row = new float[timeSteps];
                    for (int j = 0; j < timeSteps; j++)
                        row[j] = a[i + j];
                }
                
                //create features row
                a1.Add(row);
                //label row
                b1.Add(new float[] { b[i + timeSteps - 1] });
            }

            //split data into train, validation and test data set
            var xxx = splitData(a1.ToArray(), 0.1f, 0.5f);
            var yyy = splitData(b1.ToArray(), 0.1f, 0.5f);


            var retVal = new Dictionary<string, Set>();
            retVal.Add(featuresName, xxx);
            retVal.Add(labelsName, yyy);
            return retVal;
        }

        /// <summary>
        /// Build and train a RNN model.
        /// </summary>
        /// <param name="device">CPU or GPU device to train and run the model</param>
        public void Train_predict(int M, int numEpochs = 1500, int inDim = 30, int cellDim = 25, int hiDim = 5)
        {
            string featuresName = "features";
            string labelsName = "label";
            
            const int ouDim = 1;
            
            Dictionary<string, Set> dataSet = loadData(inDim, featuresName, labelsName, fun);


            var featureSet = dataSet[featuresName];
            var labelSet = dataSet[labelsName];


            ///// Debug data
            //int q = 0;
            //using (StreamWriter file = new StreamWriter("0.txt"))
            //{
            //    file.WriteLine("Train");
            //    for (int i = 0; i < featureSet.train.Length; i++)
            //    {
            //        file.Write(q + ": ");
            //        for (int j = 0; j < featureSet.train[i].Length; j++)
            //            file.Write(featureSet.train[i][j] + " ");
            //        file.Write(labelSet.train[i][0]);
            //        file.WriteLine();
            //        q++;
            //    }

            //    file.WriteLine("Valid");
            //    for (int i = 0; i < featureSet.valid.Length; i++)
            //    {
            //        file.Write(q + ": ");
            //        for (int j = 0; j < featureSet.valid[i].Length; j++)
            //            file.Write(featureSet.valid[i][j] + " ");
            //        file.Write(labelSet.valid[i][0]);
            //        file.WriteLine();
            //        q++;
            //    }

            //    file.WriteLine("Test");
            //    for (int i = 0; i < featureSet.test.Length; i++)
            //    {
            //        file.Write(q + ": ");
            //        for (int j = 0; j < featureSet.test[i].Length; j++)
            //            file.Write(featureSet.test[i][j] + " ");
            //        file.Write(labelSet.test[i][0]);
            //        file.WriteLine();
            //        q++;
            //    }
                
            //}

            // build the model

            var feature = Variable.InputVariable(new int[] { inDim + (advanced_input ? 2 : 0) }, DataType.Float, featuresName, null, false /*isSparse*/);
            var label = Variable.InputVariable(new int[] { ouDim }, DataType.Float, labelsName, new List<CNTK.Axis>() { CNTK.Axis.DefaultBatchAxis() }, false);

            var lstmModel = CreateModel(feature, ouDim, hiDim, cellDim, "timeSeriesOutput");

            Function trainingLoss = CNTKLib.SquaredError(lstmModel, label, "squarederrorLoss");
            Function prediction = CNTKLib.SquaredError(lstmModel, label, "squarederrorEval");
            
            // prepare for training
            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.0005, 1);
            TrainingParameterScheduleDouble momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(256);

            IList<Learner> parameterLearners = new List<Learner>() {
                Learner.MomentumSGDLearner(lstmModel.Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true)  };


            var trainer = Trainer.CreateTrainer(lstmModel, trainingLoss, prediction, parameterLearners);

            // train the model
            int batchSize = 20;
            int outputFrequencyInMinibatches = 50;
            int miniBatchCount = 0;

            for (int i = 1; i <= numEpochs; i++)
            {
                //get the next minibatch amount of data
                foreach (var miniBatchData in LSTMSequence.nextBatch(featureSet.train, labelSet.train, batchSize))
                {
                    var xValues = Value.CreateBatch<float>(new NDShape(1, inDim + (advanced_input ? 2 : 0)), miniBatchData.X, device);
                    var yValues = Value.CreateBatch<float>(new NDShape(1, ouDim), miniBatchData.Y, device);

                    //Combine variables and data in to Dictionary for the training
                    var batchData = new Dictionary<Variable, Value>();
                    batchData.Add(feature, xValues);
                    batchData.Add(label, yValues);

                    //train minibarch data
                    trainer.TrainMinibatch(batchData, device);

                    TestHelper.PrintTrainingProgress(trainer, miniBatchCount++, outputFrequencyInMinibatches);
                }
                
            }
            predict_test(dataSet, trainer.Model(), inDim, ouDim, batchSize, featuresName, labelsName, M);
            predict(dataSet, trainer.Model(), inDim, ouDim, batchSize, featuresName, labelsName, M);
        }


        void predict_test(Dictionary<String, Set> dataSet, Function model, int inDim, int ouDim,
            int batchSize, string featuresName, string labelsName, int M)
        {

            int len = dataSet[labelsName].test.Length;
            using (StreamWriter file = new StreamWriter("../1.txt"))
            {
                for (int i = 0; i < M; i++)
                {
                    file.WriteLine(dataSet[labelsName].test[i % len][0]);
                }

            }

            using (StreamWriter file = new StreamWriter("../2.txt"))
            {
                int sample = 0;
                foreach (var miniBatchData in nextBatch(dataSet[featuresName].test, dataSet[labelsName].test, batchSize))
                {

                    var yval = Value.CreateBatch<float>(new NDShape(1, ouDim), miniBatchData.Y, device);
                    var xval = Value.CreateBatch<float>(new NDShape(1, inDim + (advanced_input ? 2 : 0)), miniBatchData.X, device);

                    var fea = model.Arguments[0];
                    var lab = model.Output;

                    //evaluation preparation
                    var inputDataMap = new Dictionary<Variable, Value>() { { fea, xval } };
                    var outputDataMap = new Dictionary<Variable, Value>() { { lab, null } };
                    model.Evaluate(inputDataMap, outputDataMap, device);
                    var oData = outputDataMap[lab].GetDenseData<float>(lab);
                
                    foreach (var y in oData)
                    {
                        file.WriteLine(y[0]);
                        sample++;
                    }
                }
            }
        }

        void predict(Dictionary<string, Set> dataSet, Function model, int inDim, int ouDim,
            int batchSize, string featuresName, string labelsName, int n)
        {
            float res = 0;
            int len = dataSet[featuresName].test.Length;

            float[] args = new float[inDim + (advanced_input ? 2 : 0)];
            using (StreamWriter file = new StreamWriter("../3.txt"))
            {
                for (int k = 0; k < n; k++)
                {

                    if (k == 0)
                    {
                        for (int i = 0; i < dataSet[featuresName].test[0].Length; i++)
                        {
                            args[i] = dataSet[featuresName].test[0][i];
                        }
                    }
                    else
                    {
                        if (advanced_input)
                        {
                            args[0] = dataSet[featuresName].test[k % len][0];
                            args[1] = dataSet[featuresName].test[k % len][1];
                            for (int i = 0; i < inDim - 1; i++)
                            {
                                args[i + 2] = args[i + 3];
                            }
                            args[inDim + 1] = res;
                        }
                        else
                        {
                            for (int i = 0; i < inDim - 1; i++)
                            {
                                args[i] = args[i + 1];
                            }
                            args[inDim - 1] = res;
                        }
                    }
                    
                    foreach (var miniBatchData in nextBatch(new float[][] { args }, new float[][] { dataSet[labelsName].test[k % len] }, batchSize))
                    {
                        var yval = Value.CreateBatch<float>(new NDShape(1, ouDim), miniBatchData.Y, device);
                        var xval = Value.CreateBatch<float>(new NDShape(1, inDim + (advanced_input ? 2 : 0)), miniBatchData.X, device);

                        var fea = model.Arguments[0];
                        var lab = model.Output;

                        //evaluation preparation
                        var inputDataMap = new Dictionary<Variable, Value>() { { fea, xval } };
                        var outputDataMap = new Dictionary<Variable, Value>() { { lab, null } };
                        model.Evaluate(inputDataMap, outputDataMap, device);
                        var oData = outputDataMap[lab].GetDenseData<float>(lab);


                        foreach (var y in oData)
                        {
                            file.WriteLine(y[0]);
                            res = y[0];
                        }
                    }
                }
            }
        }

        Function Stabilize<ElementType>(Variable x)
        {
            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
            return CNTKLib.ElementTimes(beta, x);
        }

        Tuple<Function, Function> LSTMPCellWithSelfStabilization<ElementType>(
            Variable input, Variable prevOutput, Variable prevCellState)
        {
            int outputDim = prevOutput.Shape[0];
            int cellDim = prevCellState.Shape[0];

            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            Func<int, Parameter> createBiasParam;
            if (isFloatType)
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01f, device, "");
            else
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01, device, "");

            uint seed2 = 1;
            Func<int, Parameter> createProjectionParam = (oDim) => new Parameter(new int[] { oDim, NDShape.InferredDimension },
                    dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<int, Parameter> createDiagWeightParam = (dim) =>
                new Parameter(new int[] { dim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Function stabilizedPrevOutput = Stabilize<ElementType>(prevOutput);
            Function stabilizedPrevCellState = Stabilize<ElementType>(prevCellState);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + (createProjectionParam(cellDim) * input);

            // Input gate
            Function it =
                CNTKLib.Sigmoid(
                    (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bit = CNTKLib.ElementTimes(
                it,
                CNTKLib.Tanh(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)));

            // Forget-me-not gate
            Function ft = CNTKLib.Sigmoid(
                (Variable)(
                        projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                        CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bft = CNTKLib.ElementTimes(ft, prevCellState);

            Function ct = (Variable)bft + bit;

            // Output gate
            Function ot = CNTKLib.Sigmoid(
                (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct)));
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            Function c = ct;
            Function h = (outputDim != cellDim) ? (createProjectionParam(outputDim) * Stabilize<ElementType>(ht)) : ht;

            return new Tuple<Function, Function>(h, c);
        }


        Tuple<Function, Function> LSTMPComponentWithSelfStabilization<ElementType>(Variable input,
            NDShape outputShape, NDShape cellShape,
            Func<Variable, Function> recurrenceHookH,
            Func<Variable, Function> recurrenceHookC)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc);
            var actualDh = recurrenceHookH(LSTMCell.Item1);
            var actualDc = recurrenceHookC(LSTMCell.Item2);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            (LSTMCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });

            return new Tuple<Function, Function>(LSTMCell.Item1, LSTMCell.Item2);
        }
        

        /// <summary>
        /// Build a one direction recurrent neural network (RNN) with long-short-term-memory (LSTM) cells.
        /// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        /// </summary>
        /// <param name="input">the input variable</param>
        /// <param name="numOutputClasses">number of output classes</param>
        /// <param name="embeddingDim">dimension of the embedding layer</param>
        /// <param name="LSTMDim">LSTM output dimension</param>
        /// <param name="cellDim">cell dimension</param>
        /// <param name="device">CPU or GPU device to run the model</param>
        /// <param name="outputName">name of the model output</param>
        /// <returns>the RNN model</returns>
        
    }
}