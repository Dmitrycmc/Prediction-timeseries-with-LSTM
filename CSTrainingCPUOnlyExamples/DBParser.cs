using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.VisualBasic.FileIO;

namespace CNTK.CSTrainingExamples
{
    public class Site
    {
        public int Id { get; }
        public Dictionary<string, List<double>> data;
        public Site(int _Id, Dictionary<string, List<double>> _data)
        {
            Id = _Id;
            data = _data;
            Console.WriteLine($"Loaded {data["t"].Count} meatures from Site{Id}");
        }
    }

    public class DBParser
    {
        Site[] sites = new Site[9];

        Dictionary<string, List<double>> data_template(List<string> list)
        {
            Dictionary<string, List<double>> res = new Dictionary<string, List<double>>();
            for (int i = 0; i < list.Count; i++)
            {
                res.Add(list[i], new List<double>());
            }
            return res;
        }

        public double f (int siteId, string field, double x)
        {
        
            List<double> t = this[siteId, "t"];
            List<double> val = this[siteId, field];
        
            int i = 0;
            /// can go out of t-range
            while (t[i] < x)
            {
                i++;
            }
            if (i == 0)
            {
                i = 1;
            }
            i--;
            return val[i] + (x - t[i]) * (val[i + 1] - val[i]) / (t[i + 1] - t[i]);
        }
        



        public List<double> this[int siteId, string field]
        {
            get { return sites[siteId - 1].data[field]; }
        }

        public DBParser(string path, params string[] _list)
        {
            List<string> list = _list.ToList();
            int[] field_num = new int[list.Count];

            using (TextFieldParser tfp = new TextFieldParser(@path))
            {
                tfp.TextFieldType = FieldType.Delimited;
                tfp.SetDelimiters(";");

                Dictionary<string, List<double>> data = data_template(list);
                Dictionary<string, double> cur = new Dictionary<string, double>();

                int prev_siteId = -1, siteId = -1;

                string[] fields = tfp.ReadFields();
                for (int i = 0; i < fields.Length; i++)
                {
                    int index = list.IndexOf(fields[i]);
                    if (index != -1)
                    {
                        field_num[index] = i;
                    }
                }

                while (!tfp.EndOfData)
                {

                    fields = tfp.ReadFields();

                    siteId = Convert.ToInt32(fields[4]);

                    try
                    {
                        for (int i = 0; i < list.Count; i++)
                        {
                            string str = fields[field_num[i]];
                            cur.Add(list[i], Convert.ToDouble(str));
                        }
                    }
                    catch
                    {
                        cur = null;
                    }

                    if (siteId != prev_siteId)
                    {
                        if (prev_siteId != -1)
                        {
                            sites[prev_siteId - 1] = new Site(prev_siteId, data);
                            data = data_template(list);
                        }
                    }

                    if (cur != null)
                    {
                        for (int i = 0; i < list.Count; i++)
                        {
                            data[list[i]].Add(cur[list[i]]);
                        }
                    }
                    cur = new Dictionary<string, double>();
                    prev_siteId = siteId;
                }
                sites[siteId - 1] = new Site(siteId, data);
            }
        }
    }
}
