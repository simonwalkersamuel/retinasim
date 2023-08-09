using System;
using System.Linq;
using Vascular.Geometry;
using Vascular.Optimization;
using Vascular.Optimization.Geometric;
using Vascular.Optimization.Hybrid;
using Vascular.Optimization.Topological;
using Vascular.Structure;
using Vascular.Structure.Nodes;

namespace Retina
{
    class Optimizer
    {
        public double MaxRadiusAsymmetry { get; set; } = 5;
        public double BranchShortFraction { get; set; } = 0.1;

        public double[][] Frozen { get; set; } = null;

        public double MinTerminalLength { get; set; } = 0.0;
        public double TargetStep { get; set; } = 0.0;

        public void Standalone(Network network, IO io)
        {
            var frozen = this.Frozen?.Select(Vector3.FromArrayPermissive).ToHashSet();
            var c = io.Domain.Costs(network);

            var descent = new GradientDescentMinimizer(network)
            {
                BlockLength = 20,
                BlockRatio = 0.8,
                TargetStep = this.TargetStep
            }
            .AvoidShortTerminals(this.MinTerminalLength * 10)
            .UnfoldIfNonFinite();

            if (frozen is not null)
            {
                descent.MovingPredicate = n => !frozen.Contains(n.Parent.Branch.End.Position);
                descent.FilterByPredicate();
                // Remove all non-frozen transients
                foreach (var branch in network.Branches)
                {
                    if (!frozen.Contains(branch.End.Position) && branch.Segments.Count > 1)
                    {
                        branch.Reset();
                    }
                }
            }

            var hybrid = new HybridMinimizer(network)
            {
                GeometryIterations = 100,
                MinTerminalLength = this.MinTerminalLength,
                Minimizer = descent,
                RemoveTransients = false,
                ResetStrideAlways = true
            }
            .AddHierarchicalCosts(c, false);

            hybrid.Iterate(0);
            network.Set(true, true, true);
        }

        public void Trim(Network network, Action<Terminal> onTrim = null)
        {
            if (this.MaxRadiusAsymmetry > 0)
            {
                network.Set(true, true);
                var asy = new AsymmetryRemover()
                {
                    OnCull = onTrim,
                    RadiusRatio = this.MaxRadiusAsymmetry,
                    FlowRatio = double.PositiveInfinity
                };
                asy.Act(network.Root);
                network.Set(true, true);
            }
        }

        public void Optimize(Network network, HierarchicalCosts costs,
            double targetStride, double shortLength,
            double L0, double Q0, Action<Terminal> onTrim,
            Func<INode, bool> predicate = null, bool preserveTransients = false)
        {
            var descent = new GradientDescentMinimizer(network)
            {
                TargetStep = targetStride,
                BlockLength = 20,
                BlockRatio = 0.8,
            }
            .AvoidShortTerminals(shortLength)
            .UnfoldIfNonFinite();

            var hybrid = new HybridMinimizer(network)
            {
                GeometryIterations = 100,
                MinTerminalLength = 0.1 * shortLength,
                Minimizer = descent,
                RemoveTransients = !preserveTransients,
                ResetStrideAlways = true,
                OnTrim = onTrim
            }
            .AddHierarchicalCosts(costs, false);

            if (predicate is not null)
            {
                descent.OnGradientComputed += G =>
                {
                    foreach (var (n, g) in G)
                    {
                        if (!predicate(n))
                        {
                            g.Copy(Vector3.ZERO);
                        }
                    }
                };
                hybrid.ActionPredicate = A =>
                {
                    return A.A.Nodes.All(predicate)
                        && A.B.Nodes.All(predicate);
                };
            }

            var iterations = 0;
            if (this.BranchShortFraction != 0 && L0 != 0)
            {
                var L1 = L0 / Math.Pow(Q0, 1.0 / 3.0);
                hybrid
                    .AddDistanceEvaluation()
                    .AddRegroup(L1, this.BranchShortFraction, true, true);
                iterations = 4;
            }

            hybrid.Iterate(iterations);
        }
    }
}
