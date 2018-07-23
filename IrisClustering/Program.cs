﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Threading.Tasks;

namespace IrisClustering
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        static async Task Main(string[] args)
        {
            PredictionModel<IrisData, ClusterPrediction> model = Train();

            await model.WriteAsync(_modelPath);

            var prediction = model.Predict(TestIrisData.Setosa);

            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }

        private static PredictionModel<IrisData, ClusterPrediction> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_dataPath).CreateFrom<IrisData>(separator: ','),

                new ColumnConcatenator(
                "Features",
                "SepalLength",
                "SepalWidth",
                "PetalLength",
                "PetalWidth"),

                new KMeansPlusPlusClusterer() { K = 3 }
            };

            var model = pipeline.Train<IrisData, ClusterPrediction>();

            return model;

        }
    }
}
