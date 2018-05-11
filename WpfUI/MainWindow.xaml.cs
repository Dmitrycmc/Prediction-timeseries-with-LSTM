using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using System.IO;
using System.ComponentModel;
using Microsoft.Win32;

namespace WpfUI
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            for (int i= 1; i <= 9; i++)
            {
                comboBox.Items.Add(i);
            }
            comboBox.SelectedIndex = 0;

            button_show_Click(this, null);

        }

        private void button_start_Click(object sender, RoutedEventArgs e)
        {
            string siteId = comboBox.SelectedItem.ToString();
            string M = textBox_M.Text;
            string numEpochs = textBox_numEpochs.Text;
            string inDim = textBox_inDim.Text;
            string cellDim = textBox_cellDim.Text;
            string hiDim = textBox_hiDim.Text;
            bool advanced_input = (bool)checkBox_advanced_input.IsChecked;

            string cmdargs = siteId + " " + (advanced_input ? "1": "0") + " " + M + " " + numEpochs + " " + inDim + " " + cellDim + " " + hiDim;
            System.Diagnostics.Process.Start("CNTKCSTrainingCPUOnlyExamples.exe", cmdargs);
        }

        private void button_show_Click(object sender, RoutedEventArgs e)
        {
            List<double> original, predict1, predict;


            original = new List<double>();
            predict1 = new List<double>();
            predict = new List<double>();

            try
            {
                using (StreamReader file = new StreamReader("../1.txt"))
                {
                    string str;
                    double val = 0;
                    while ((str = file.ReadLine()) != null)
                    {
                        val = Convert.ToDouble(str);
                        original.Add(val);
                    }
                }
            }
            catch
            {
                MessageBox.Show("Ogirinal data not found.");
            }
            try
            {
                using (StreamReader file = new StreamReader("../2.txt"))
                {
                    string str;
                    double val = 0;
                    while ((str = file.ReadLine()) != null)
                    {
                        val = Convert.ToDouble(str);
                        predict1.Add(val);
                    }
                }
            }
            catch
            {
                MessageBox.Show("Preict1 data not found.");
            }

            try
            {
                using (StreamReader file = new StreamReader("../3.txt"))
                {
                    string str;
                    double val = 0;
                    while ((str = file.ReadLine()) != null)
                    {
                        val = Convert.ToDouble(str);
                        predict.Add(val);
                    }
                }
            }

            catch
            {
                MessageBox.Show("Predict data not found.");
            }

            double squared_error = 0;
            List<double> error = new List<double>();
            for (int i = 0; i < Math.Min(original.Count, predict.Count); i++)
            {
                double err = Math.Abs(original[i] - predict[i]);
                error.Add(err);
                squared_error += err * err;
            }

            squared_error = Math.Sqrt(squared_error / Math.Min(original.Count, predict.Count));
            textBox_squared_error.Text = squared_error.ToString();



            Wpf.CartesianChart.PointShapeLine.PointShapeLineExample plot = 
                new Wpf.CartesianChart.PointShapeLine.PointShapeLineExample(original, predict1, predict, error);
            plot.Width = stack_panel.Width;
            plot.Height = stack_panel.Height;
            if (stack_panel.Children.Count != 0)
            {
                stack_panel.Children.Clear();
            } 
            stack_panel.Children.Add(plot);

            //// error ocenca
        }

        private void button_save_plot_Click(object sender, RoutedEventArgs e)
        {
            RenderTargetBitmap rtb = new RenderTargetBitmap((int)stack_panel.ActualWidth, (int)stack_panel.ActualHeight, 96, 96, PixelFormats.Pbgra32);
            rtb.Render(stack_panel.Children[0]);

            PngBitmapEncoder png = new PngBitmapEncoder();
            png.Frames.Add(BitmapFrame.Create(rtb));
            MemoryStream stream = new MemoryStream();
            png.Save(stream);
            
            SaveFileDialog dlg = new SaveFileDialog();
            dlg.Filter = "PNG images(*.png) | *.png";
            //dlg.InitialDirectory = Environment.CurrentDirectory;

            if (dlg.ShowDialog() == true)
            {
                using (FileStream fs = new FileStream(dlg.FileName, FileMode.Create))
                {
                    stream.WriteTo(fs);
                }
            }
        }

        private void button_save_window_Click(object sender, RoutedEventArgs e)
        {
            RenderTargetBitmap rtb = new RenderTargetBitmap((int)mainWindow.ActualWidth, (int)mainWindow.ActualHeight, 96, 96, PixelFormats.Pbgra32);
            rtb.Render(mainWindow);

            PngBitmapEncoder png = new PngBitmapEncoder();
            png.Frames.Add(BitmapFrame.Create(rtb));
            MemoryStream stream = new MemoryStream();
            png.Save(stream);
            
            stream.Position = 0;

            SaveFileDialog dlg = new SaveFileDialog();
            dlg.Filter = "PNG images(*.png) | *.png";
            //dlg.InitialDirectory = Environment.CurrentDirectory;

            if (dlg.ShowDialog() == true)
            {
                using (FileStream fs = new FileStream(dlg.FileName, FileMode.Create))
                {
                    stream.WriteTo(fs);
                }
            }
            
            
        }
    }
}
