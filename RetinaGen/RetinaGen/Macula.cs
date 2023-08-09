using System;
using System.Collections.Generic;
using Vascular.Construction.LSV;
using Vascular.Construction.LSV.Defaults;
using Vascular.Geometry;
using Vascular.Geometry.Generators;
using Vascular.Optimization;
using Vascular.Structure;
using Vascular.Structure.Actions;
using Vascular.Structure.Nodes;

namespace Retina
{
    class Macula
    {
        public double FlowFactor { get; set; } = 1;
        public double SpacingMax { get; set; } = 0.3e3;
        public double SpacingMin { get; set; } = 0.3e3;
        public int Refinements { get; set; } = 0;
        public double Sparsity { get; set; } = 0.5;
        public double MobileRangeFactor { get; set; } = 1.0;
        public bool OptimizeTopology { get; set; } = false;
        public double MinAlignment { get; set; } = -0.7;

        public void Grow(Network network, Domain domain, Optimizer optimizer, Random random)
        {
            var costs = domain.Costs(network);

            var inMacula = domain.MaculaFoveaPredicate();
            var inDomain = domain.DomainPredicate();
            var isMovable = domain.FoveaDistancePredicate(domain.MaculaRadius * this.MobileRangeFactor);

            var spacings = Utility.Spacings(this.SpacingMin, this.SpacingMax, this.Refinements);
            var states = new List<LatticeState>(this.Refinements + 1);
            foreach (var spacing in spacings)
            {
                var lattice = Utility.Lattice(spacing, network);
                var state = new LatticeState(network, lattice)
                {
                    TerminalPairCostFunction = (T, t) =>
                    {
                        var W = T.Flow + t.Flow;
                        var p = W * T.Upstream.Start.Position + T.Flow * T.Position + t.Flow * t.Position;
                        return ActionEstimates.CreateBifurcation(costs, T, T.Upstream, p / (2 * W));
                    },
                    BifurcationPositionFunction = Spread.NearestPointPosition(),
                    ExteriorPredicate = (z, x) => inMacula(x) && inDomain(z, x),
                    ExteriorOrderingGenerator = RandomOrdering.PermuteExterior(random),
                    InteriorMode = InteriorMode.Default,
                    BeforeSpreadAction = () => costs.SetCache(),
                };
                Refinement.SetFlowByDeterminant(domain.FlowRateDensity * this.FlowFactor, state);
                state.AfterSpreadAction += () =>
                    optimizer.Optimize(network, costs,
                        this.SpacingMax * 0.1, this.SpacingMin * 0.1,
                        0.0, 0.0, t => state.Remove(t, false), n => isMovable(n.Position));

                var singleBuild = new SingleBuild(state);
                var foveaDir = Vector3.FromArrayPermissive(domain.FoveaPosition) - Vector3.FromArrayPermissive(domain.OpticDiscPosition);
                foveaDir = foveaDir.Normalize();
                state.TerminalPairPredicate = (T, t) =>
                {
                    if (!singleBuild.Predicate(T, t))
                    {
                        return false;
                    }
                    var tdir = t.Position - T.Upstream.Start.Position;
                    return tdir * foveaDir / tdir.Length >= this.MinAlignment;
                };
                state.BeforeSpreadAction += singleBuild.BeforeSpread;
                state.AfterSpreadAction += singleBuild.AfterSpread;
                state.TerminalPairBuildAction += singleBuild.OnBuild;

                if (this.OptimizeTopology)
                {
                    state.OnExit += () => optimizer.Optimize(network, costs,
                        this.SpacingMax * 0.1, this.SpacingMin * 0.1,
                        spacing, domain.FlowRateDensity * this.FlowFactor * lattice.Determinant, t => state.Remove(t, false),
                        n => isMovable(n.Position));
                }
                state.OnExit += () => Utility.ResolveInternal(network, minPerturbation: this.SpacingMin * 0.01);

                var cRand = new CylindricalRandom(random, spacing / 2, 0);
                state.TerminalConstructor = (x, Q) => new Terminal(x + cRand.NextVector3(), Q);

                states.Add(state);
            }

            if (this.Sparsity != 1.0)
            {
                var S = states[^1];
                S.OnExit += () =>
                {
                    foreach (var (z, T) in S.SingleInterior)
                    {
                        if (inMacula(T.Position) && random.NextDouble() >= this.Sparsity)
                        {
                            Topology.CullTerminal(T);
                        }
                    }
                    network.Set(true, true);

                    optimizer.Optimize(network, costs,
                        this.SpacingMax * 0.1, this.SpacingMin * 0.1,
                        0.0, 0.0, null, n => isMovable(n.Position));
                };
            }

            var sequence = new LatticeSequence()
            {
                Elements = states
            };
            sequence.Initialize();
            sequence.Complete();
        }
    }
}