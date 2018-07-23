using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace IrisClustering
{
    class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;
    }

    class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
