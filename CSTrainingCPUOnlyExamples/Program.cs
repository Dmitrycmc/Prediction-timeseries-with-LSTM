using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CSTrainingExamples
{
    class Program
    {
        static string path = @"../plankton.csv";
        static DBParser b = new DBParser(
            path, 
            "t", "lpC", "MLD", "light"
        );
        
        static void Main(string[] args)
        {
            int siteId = Convert.ToInt32(args[0]);
            Console.WriteLine("SiteId = {0}", siteId);

            bool advanced_input = (args[1] == "1");
            Console.WriteLine("Advanced input: {0}", advanced_input);

            int M = Convert.ToInt32(args[2]);
            Console.WriteLine("M = {0}", M);

            int numEpochs = Convert.ToInt32(args[3]);
            Console.WriteLine("numEpochs = {0}", numEpochs);
            
            int inDim = Convert.ToInt32(args[4]);
            Console.WriteLine("inDim = {0}", inDim);

            int cellDim = Convert.ToInt32(args[5]);
            Console.WriteLine("cellDim = {0}", cellDim);
            
            int hiDim = Convert.ToInt32(args[6]);
            Console.WriteLine("hidim = {0}", hiDim);

            DeviceDescriptor device = DeviceDescriptor.CPUDevice;
            Console.WriteLine($"======== running LSTMSequence.Train using {DeviceDescriptor.CPUDevice} ========");

            LSTMSequence network = new LSTMSequence(b, siteId, device, advanced_input);
            network.Train_predict(M, numEpochs, inDim, cellDim, hiDim);
        }
    }
}
