using System;
using System.Collections.Generic;
using System.Linq;
using Vascular.Geometry;
using Vascular.Geometry.Graphs;
using Vascular.Structure;
using Vascular.Structure.Nodes;

namespace Vascular.IO.Amira
{
    public class AmiraData
    {
        public double[][] VertexCoordinates { get; set; }
        public int[][] EdgeConnectivity { get; set; }
        public int[] NumEdgePoints { get; set; }
        public double[][] EdgePointCoordinates { get; set; }
        public double[] Radius { get; set; }

        public void Convert(IReadOnlyList<Branch> branches)
        {
            var nodes = GetNodeDict(branches);
            this.VertexCoordinates = new double[nodes.Count][];
            foreach (var kv in nodes)
            {
                this.VertexCoordinates[kv.Value] = kv.Key.Position.ToArray();
            }

            this.EdgeConnectivity = new int[branches.Count][];
            this.NumEdgePoints = new int[branches.Count];
            for (var i = 0; i < branches.Count; ++i)
            {
                this.EdgeConnectivity[i] = new int[2]
                {
                    nodes[branches[i].Start],
                    nodes[branches[i].End]
                };
                this.NumEdgePoints[i] = branches[i].Segments.Count + 1;
            }

            var nPoints = this.NumEdgePoints.Sum();
            this.EdgePointCoordinates = new double[nPoints][];
            this.Radius = new double[nPoints];
            var offset = 0;
            for (var i = 0; i < branches.Count; ++i)
            {
                foreach (var node in branches[i].Nodes)
                {
                    this.EdgePointCoordinates[offset] = node.Position.ToArray();
                    this.Radius[offset] = node.MaxRadius();
                }
            }
        }

        private static Dictionary<BranchNode, int> GetNodeDict(IEnumerable<Branch> branches)
        {
            var nodes = new Dictionary<BranchNode, int>();
            foreach (var b in branches)
            {
                TryAddNode(nodes, b.Start);
                TryAddNode(nodes, b.End);
            }
            return nodes;
        }

        private static void TryAddNode(Dictionary<BranchNode, int> nodes, BranchNode node)
        {
            if (!nodes.ContainsKey(node))
            {
                nodes[node] = nodes.Count;
            }
        }

        public Network Convert()
        {
            var VG = VertexGraph();
            if (VG.E.Count + 1 != VG.V.Count)
            {
                throw new InputException($"Graph is not tree: V = {VG.V.Count}, E = {VG.E.Count}, expected V = E + 1");
            }
            var R = PointIndices(VG);
            var (rootEdge, rootThickness) = MostThick(R);
            var source = Root(rootEdge, rootThickness);
            if (source is null)
            {
                return FromSegments(Segments());
            }
            var (EG, T) = EdgeGraph(R);
            var sourceVertex = EG.V[source.Position];
            var sourceEdge = sourceVertex.E.First.Value;
            var rootSegment = new Segment() { Start = source };
            Fill(rootSegment, EG, T, sourceEdge, sourceEdge.OtherSafe(sourceVertex));
            source.Child = rootSegment;
            var N = new Network()
            {
                Source = source
            };
            foreach (var n in N.BranchNodes)
            {
                n.Network = N;
            }
            return N;
        }

        public static Network FromSegments(IEnumerable<Segment> S)
        {
            var (G, R) = EdgeGraphFromSegments(S);
            if (G.E.Count + 1 != G.V.Count)
            {
                throw new InputException($"Graph is not tree: V = {G.V.Count}, E = {G.E.Count}, expected V = E + 1");
            }
            var sourceEdge = G.E.Values
                .Where(e => e.S.E.Count == 1 || e.E.E.Count == 1)
                .ArgMin(e => -R[e]);
            var sourceVertex = sourceEdge.S.E.Count == 1 ? sourceEdge.S : sourceEdge.E;
            var source = new RadiusSource(sourceVertex.P, R[sourceEdge]);
            var rootSegment = new Segment() { Start = source };
            Fill(rootSegment, G, R, sourceEdge, sourceEdge.OtherSafe(sourceVertex));
            source.Child = rootSegment;
            var N = new Network()
            {
                Source = source
            };
            foreach (var n in N.BranchNodes)
            {
                n.Network = N;
            }
            return N;
        }

        public IEnumerable<Network> ConvertMultiple()
        {
            var VG = VertexGraph();
            var R = PointIndices(VG);
            var (EG, T) = EdgeGraph(R);
            var visited = new HashSet<Vertex>(EG.V.Count);

            foreach (var v in VG.V)
            {
                if (visited.Contains(v.Value))
                {
                    continue;
                }

                var (SE, SV) = EG.ConnectedComponent(v.Value);
                visited.UnionWith(SV);
                var SR = SE.ToDictionary(e => e, e => R[e]);

                var (rootEdge, rootThickness) = MostThick(SR);
                var source = Root(rootEdge, rootThickness);
                var sourceVertex = EG.V[source.Position];
                var sourceEdge = sourceVertex.E.First.Value;
                var rootSegment = new Segment() { Start = source };
                Fill(rootSegment, EG, T, sourceEdge, sourceEdge.OtherSafe(sourceVertex));
                source.Child = rootSegment;
                var N = new Network()
                {
                    Source = source
                };
                foreach (var n in N.BranchNodes)
                {
                    n.Network = N;
                }
                yield return N;
            }
        }

        public IEnumerable<Segment> Segments()
        {
            var VG = VertexGraph();
            var R = PointIndices(VG);
            var (_, T) = EdgeGraph(R);
            foreach (var (e, t) in T)
            {
                yield return new Segment()
                {
                    Start = new Dummy() { Position = e.S.P },
                    End = new Dummy() { Position = e.E.P },
                    Radius = t / 2
                };
            }
        }

        public static void Fill(Segment segment, Graph EG, Dictionary<Edge, double> T, Edge edge, Vertex to)
        {
            var bStart = segment.Start as BranchNode;
            while (to.E.Count == 2)
            {
                var tr = new Transient()
                {
                    Parent = segment,
                    Child = new Segment(),
                    Position = to.P
                };
                tr.Child.Start = tr;
                segment.End = tr;
                segment.Radius = T[edge];
                // Iterate
                segment = tr.Child;
                edge = to.E.First(e => e != edge);
                to = edge.OtherSafe(to);
            }

            if (to.E.Count == 1)
            {
                var term = new Terminal(to.P, 1)
                {
                    Parent = segment
                };
                segment.End = term;
            }
            else
            {
                // Create bifurcations
                var children = to.E.Where(e => e != edge).ToArray();
                var (bifurc, childSegs, bifurcs) = MakeBifurcationGroup(children, to.P);
                for (var i = 0; i < children.Length; ++i)
                {
                    Fill(childSegs[i], EG, T, children[i], children[i].OtherSafe(to));
                }
                segment.End = bifurc;
                bifurc.Parent = segment;
                foreach (var bf in bifurcs)
                {
                    bf.UpdateDownstream();
                }
            }
            segment.Radius = T[edge];
            // Create branch & init
            var bEnd = segment.End as BranchNode;
            var branch = new Branch() { Start = bStart, End = bEnd };
            branch.Initialize();
        }

        public static (Bifurcation, Segment[], IEnumerable<Bifurcation>) MakeBifurcationGroup(Edge[] C, Vector3 P)
        {
            var S = C.Select(e => new Segment()).ToArray();
            var B = new Bifurcation() { Position = P };
            var BB = new List<Bifurcation>(S.Length - 1);
            var b = B;
            for (var i = 0; i < C.Length - 2; ++i)
            {
                BB.Add(b);
                b.Children[0] = S[i];
                S[i].Start = b;
                var be = new Bifurcation() { Position = P };
                b.Children[1] = new Segment() { Start = b };
                b.Children[1].End = be;
                be.Parent = b.Children[1];
                var br = new Branch() { Start = b, End = be };
                br.Initialize();
                b = be;
            }
            BB.Add(b);
            b.Children[0] = S[^2];
            S[^2].Start = b;
            b.Children[1] = S[^1];
            S[^1].Start = b;
            return (B, S, BB);
        }

        public Graph VertexGraph()
        {
            var G = new Graph();
            foreach (var ec in this.EdgeConnectivity)
            {
                var p0 = Position(ec[0], this.VertexCoordinates);
                var p1 = Position(ec[1], this.VertexCoordinates);
                var v0 = G.AddVertex(p0);
                var v1 = G.AddVertex(p1);
                var e = new Edge(v0, v1);
                G.AddEdge(e);
            }
            return G;
        }

        //public IEnumerable<Graph> Decompose(Graph G)
        //{
        //    var V = new HashSet<Vertex>(G.V.Count);
        //    foreach (var (p, v) in G.V)
        //    {
        //        if (V.Contains(v))
        //        {
        //            continue;
        //        }

        //        var g = new Graph();

        //    }
        //}

        public Dictionary<Edge, Range> PointIndices(Graph G)
        {
            var o = 0;
            var R = new Dictionary<Edge, Range>(G.E.Count);
            for (var i = 0; i < this.NumEdgePoints.Length; ++i)
            {
                var r = new Range(o, o + this.NumEdgePoints[i]);
                var p0 = Position(r.Start.Value, this.EdgePointCoordinates);
                var p1 = Position(r.End.Value - 1, this.EdgePointCoordinates);
                var v0 = G.V[p0];
                var v1 = G.V[p1];
                var e = new Edge(v0, v1);
                R[G.E[e]] = r;
                o += this.NumEdgePoints[i];
            }
            return R;
        }

        public (Graph G, Dictionary<Edge, double> T) EdgeGraph(Dictionary<Edge, Range> R)
        {
            var G = new Graph();
            var t = new Dictionary<Edge, double>();
            foreach (var r in R.Values)
            {
                var P = this.EdgePointCoordinates[r];
                var T = this.Radius[r];
                for (var i = 0; i < P.Length - 1; ++i)
                {
                    var s = G.AddVertex(Position(i, P));
                    var e = G.AddVertex(Position(i + 1, P));
                    var E = new Edge(s, e);
                    G.AddEdge(E);
                    t[E] = (T[i] + T[i + 1]) * 0.5;
                }
            }
            return (G, t);
        }

        public static (Graph G, Dictionary<Edge, double> R) EdgeGraphFromSegments(IEnumerable<Segment> S)
        {
            var G = new Graph();
            var R = new Dictionary<Edge, double>();
            foreach (var s in S)
            {
                var a = G.AddVertex(s.Start.Position);
                var b = G.AddVertex(s.End.Position);
                var e = G.AddEdge(new(a, b));
                R[e] = s.Radius;
            }
            return (G, R);
        }

        public (Edge e, double r) MostThick(Dictionary<Edge, Range> R)
        {
            // Average as writing out will make children of root have one node with radius equal to root
            R.ArgMin(kv => -this.Radius[kv.Value].Average(), out var kv, out var t);
            return (kv.Key, -t);
        }

        public static Source Root(Edge R, double r)
        {
            if (R.S.E.Count == 1)
            {
                return new RadiusSource(R.S.P, r);
            }
            else if (R.E.E.Count == 1)
            {
                return new RadiusSource(R.E.P, r);
            }
            return null;
        }

        private static Vector3 Position(int i, double[][] p)
        {
            return new Vector3(p[i][0], p[i][1], p[i][2]);
        }
    }
}
