using System;
using LinearRegres;

namespace Regres
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("Укажите путь к файлу");
            var path = Console.ReadLine();
            Console.WriteLine("Укажите размерность прстранства");
            var dimension =Convert.ToInt32(Console.ReadLine());
            var ff = new FileFramework(@"E:\Users\Nikita\Documents\Visual Studio 2013\Projects\NSUBigData\LinearRegres\iris.csv", Convert.ToInt32(dimension));
           
            Console.WriteLine("\nBegin Logistic Regression (binary) Classification demo");
            Console.WriteLine("Goal is to demonstrate training using gradient descent");

            var numFeatures = dimension-1; 
            var numRows = 100;
            var seed = 1;

            Console.WriteLine("\nGenerating " + numRows +
                              " artificial data items with " + numFeatures + " features");
            var allData = ff.ParseFile();

            Console.WriteLine("Creating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, 0, out trainData, out testData);
            Console.WriteLine("Done");


            Console.WriteLine("\nTraining data: \n");
            ShowData(trainData, 3, 2, true);

            Console.WriteLine("\nTest data: \n");
            ShowData(testData, 3, 2, true);


            Console.WriteLine("Creating LR binary classifier");
            var lc = new LogisticClassifier(numFeatures); 

            var maxEpochs = 1000;
            Console.WriteLine("Setting maxEpochs = " + maxEpochs);
            var alpha = 0.01;
            Console.WriteLine("Setting learning rate = " + alpha.ToString("F2"));

            Console.WriteLine("\nStarting training using (stochastic) gradient descent");
            double[] weights = lc.Train(trainData, maxEpochs, alpha);
            Console.WriteLine("Training complete");

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 4, true);

            double trainAcc = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " +
                              trainAcc.ToString("F4"));

            double testAcc = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " +
                              testAcc.ToString("F4"));

            Console.WriteLine("\nEnd LR binary classification demo\n");
            Console.ReadLine();
        }

        

        private static void MakeTrainTest(double[][] allData, int seed,
            out double[][] trainData, out double[][] testData)
        {
            var rnd = new Random(seed);
            var totRows = allData.Length;
            var numTrainRows = (int) (totRows*0.80); // 80% hard-coded
            var numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            var copy = new double[allData.Length][]; // ref copy of all data
            for (var i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (var i = 0; i < copy.Length; ++i) // scramble order
            {
                var r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (var i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (var i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // MakeTrainTest


        public static void ShowData(double[][] data, int numRows,
            int decimals, bool indices)
        {
            var len = data.Length.ToString().Length;
            for (var i = 0; i < numRows; ++i)
            {
                if (indices)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (var j = 0; j < data[i].Length; ++j)
                {
                    var v = data[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine(". . .");
            var lastRow = data.Length - 1;
            if (indices)
                Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
            for (var j = 0; j < data[lastRow].Length; ++j)
            {
                var v = data[lastRow][j];
                if (v >= 0.0)
                    Console.Write(" "); // '+'
                Console.Write(v.ToString("F" + decimals) + "  ");
            }
            Console.WriteLine("\n");
        }

        private static void ShowVector(double[] vector, int decimals, bool newLine)
        {
            foreach (var t in vector)
                Console.Write(t.ToString("F" + decimals) + " ");
            Console.WriteLine("");
            if (newLine)
                Console.WriteLine("");
        }

    }
}
