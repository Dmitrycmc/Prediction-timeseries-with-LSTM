using System;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using System.Collections.Generic;

namespace Wpf.CartesianChart.PointShapeLine
{
    public partial class PointShapeLineExample : UserControl
    {
        public PointShapeLineExample(List<double> original, List<double> predict1, List<double> predict, List<double> error)
        {
            InitializeComponent();

            SeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Prediction",
                    Values = new ChartValues<double>(predict),
                    LineSmoothness = 0,
                    PointGeometry = null,
                },

                new LineSeries
                {
                    Title = "Original",
                    Values = new ChartValues<double>(original),
                    LineSmoothness = 0,
                    PointGeometry = null
                },

                new LineSeries
                {
                    Title = "Prediction M = 1",
                    Values = new ChartValues<double>(predict1),
                    LineSmoothness = 0,
                    PointGeometry = null,
                },

                new LineSeries
                {
                    Title = "Abs of error",
                    Values = new ChartValues<double>(error),
                    LineSmoothness = 0,
                    PointGeometry = null,
                },
            };

            
            YFormatter = value => value.ToString();
            
            DataContext = this;
        }

        public SeriesCollection SeriesCollection { get; set; }
        public Func<double, string> YFormatter { get; set; }

    }
}