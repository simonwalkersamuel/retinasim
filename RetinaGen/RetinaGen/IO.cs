using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Vascular.Geometry;
using Vascular.IO.Amira;
using Vascular.IO.Text;
using Vascular.Structure;
using Vascular.Structure.Nodes;
using Vascular.Structure.Splitting;

namespace Retina
{
    class IO
    {
        public string ArteryInPath { get; set; } = "";
        public string VeinInPath { get; set; } = "";
        public string ArteryOutPath { get; set; } = "";
        public string VeinOutPath { get; set; } = "";

        public string Actions { get; set; } = "";
        public int Seed { get; set; } = -1;

        public Domain Domain { get; set; } = new();
        public Major Major { get; set; } = new();
        public Macula Macula { get; set; } = new();
        public Interleaving Interleaving { get; set; } = new();
        public Optimizer Optimizer { get; set; } = new();

        public Random Random()
        {
            return this.Seed < 0 ? new() : new(this.Seed);
        }

        public static void WriteDefaults(string path = "example.json")
        {
            var opts = JsonSerializerOptions;
            File.WriteAllText(path, JsonSerializer.Serialize(new IO(), opts));
        }

        public static JsonSerializerOptions JsonSerializerOptions => new()
        {
            WriteIndented = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals | JsonNumberHandling.AllowReadingFromString,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };

        public void EnsureOutputPaths()
        {
            if (string.IsNullOrWhiteSpace(this.ArteryOutPath))
            {
                this.ArteryOutPath = this.ArteryInPath + ".csv";
            }
            if (string.IsNullOrWhiteSpace(this.VeinOutPath) && !string.IsNullOrWhiteSpace(this.VeinInPath))
            {
                this.VeinOutPath = this.VeinInPath + ".csv";
            }
        }

        public Network[] Read()
        {
            if (string.IsNullOrWhiteSpace(this.ArteryInPath))
            {
                return Create();
            }

            var A = Read(this.ArteryInPath);
            A.Splitting = new Murray() { Exponent = this.Domain.ArteryMurrayExponent };
            if (string.IsNullOrWhiteSpace(this.VeinInPath))
            {
                A.Set(true, true);
                return new[] { A };
            }

            var V = Read(this.VeinInPath);
            V.Splitting = new Murray() { Exponent = this.Domain.VeinMurrayExponent };
            EnsureOffset(A, V);
            A.Set(true, true);
            V.Set(true, true);
            V.Output = true;
            return new[] { A, V };
        }

        private Network[] Create()
        {
            var nets = new List<Network>();
            if (string.IsNullOrWhiteSpace(this.ArteryOutPath))
            {
                throw new Exception("No artery source or target specified: terminating");
            }
            nets.Add(new Network()
            {
                Source = new RadiusSource(new(), this.Domain.ArteryRootRadius),
                Splitting = new Murray() { Exponent = this.Domain.ArteryMurrayExponent }
            });
            if (!string.IsNullOrWhiteSpace(this.VeinOutPath))
            {
                nets.Add(new Network()
                {
                    Source = new RadiusSource(new(), this.Domain.VeinRootRadius),
                    Splitting = new Murray() { Exponent = this.Domain.VeinMurrayExponent },
                    Output = true
                });
                EnsureOffset(nets[0], nets[1]);
            }
            return nets.ToArray();
        }

        public void Write(Network[] N)
        {
            EnsureOutputPaths();
            if (N.Length == 2)
            {
                N[1].Set(true, true, true);
                Write(this.VeinOutPath, N[1]);
            }
            N[0].Set(true, true, true);
            Write(this.ArteryOutPath, N[0]);
        }

        private static void EnsureOffset(Network arterial, Network venous)
        {
            var required = (arterial.Source.RootRadius + venous.Source.RootRadius) * 2;
            var offset = arterial.Source.Position - venous.Source.Position;
            var distance = offset.Length;

            if (distance < required)
            {
                var remaining = required - distance;
                var push = remaining * 0.5 * (offset.NormalizeSafe() ?? Vector3.UNIT_X);
                arterial.Source.SetPosition(arterial.Source.Position + push);
                venous.Source.SetPosition(venous.Source.Position - push);
            }
        }

        public static Network Read(string path)
        {
            var ext = Path.GetExtension(path);
            switch (ext)
            {
                case ".json":
                    return JsonSerializer.Deserialize<AmiraData>(File.ReadAllText(path)).Convert();
                case ".csv":
                    using (var reader = new StreamReader(new FileStream(path, FileMode.Open, FileAccess.Read)))
                    {
                        return AmiraData.FromSegments(SegmentCsv.Read(reader));
                    }
                default:
                    throw new Exception($"File type not supported: '{ext}'");
            }
        }

        public static void Write(string path, Network network)
        {
            var ext = Path.GetExtension(path);
            switch (ext)
            {
                case ".json":
                    var amData = new AmiraData();
                    amData.Convert(network.Branches.ToList());
                    File.WriteAllText(path, JsonSerializer.Serialize(amData));
                    break;
                case ".csv":
                    using (var writer = new StreamWriter(new FileStream(path, FileMode.Create, FileAccess.Write)))
                    {
                        SegmentCsv.Write(writer, network.Segments);
                    }
                    break;
                default:
                    throw new Exception($"File type not supported: '{ext}'");
            }
        }
    }
}
