using System;
using System.Collections.Generic;
using System.Linq;
using Vascular.Construction.ACCO;
using Vascular.Construction.LSV;
using Vascular.Construction.LSV.Defaults;
using Vascular.Geometry;
using Vascular.Geometry.Generators;
using Vascular.Geometry.Lattices.Manipulation;
using Vascular.Optimization;
using Vascular.Structure;
using Vascular.Structure.Actions;
using Vascular.Structure.Nodes;
using Vascular.Structure.Nodes.Pinned;

namespace Retina
{
    class Major
    {
        public double SpacingMax { get; set; } = 4e3;
        public double SpacingMin { get; set; } = 0.5e3;
        public int Refinements { get; set; } = 3;

        public bool RemoveStartingLeaves { get; set; } = true;
        public double PreSpacing { get; set; } = 0;

        public bool ClearMacula { get; set; } = true;
        public double TerminalDeviation { get; set; } = 0;
        public bool TerminalsMobile { get; set; } = false;

        public double FrozenFactor { get; set; } = 0.0;

        public bool UpdateFrozen { get; set; } = false;
        public bool FreezeBranches { get; set; } = false;

        public double TopologyOptimizationLimit { get; set; } = 0.0;

        public void Grow(Network network, Domain domain, Optimizer optimizer, Random random)
        {
            InitialCCO(network, domain, optimizer, random, out var startingLeaves);

            var costs = domain.Costs(network);
            var states = new List<LatticeState>(this.Refinements + 1);
            var inDomain = domain.DomainPredicate();
            var inMacula = domain.MaculaPredicate();
            var maculaRegion = domain.MaculaRegion();

            var exteriorPredicate = this.ClearMacula
                ? (z, x) => inDomain(z, x) && !inMacula(x)
                : inDomain;

            var strides = Utility.Spacings(this.SpacingMin, this.SpacingMax, this.Refinements);
            var frozen = optimizer.Frozen?.Select(Vector3.FromArrayPermissive).ToHashSet();
            foreach (var stride in strides)
            {
                var lattice = Utility.Lattice(stride, network);
                var state = new LatticeState(network, lattice)
                {
                    TerminalPairCostFunction = (T, t) =>
                    {
                        var W = T.Flow + t.Flow;
                        var p = W * T.Upstream.Start.Position + T.Flow * T.Position + t.Flow * t.Position;
                        return ActionEstimates.CreateBifurcation(costs, T, T.Upstream, p / (2 * W));
                    },
                    BifurcationPositionFunction = Spread.NearestPointPosition(),
                    BifurcationSegmentSelector = (b, t) => Spread.NearestSegmentSelector(b, t),
                    ExteriorPredicate = exteriorPredicate,
                    ExteriorOrderingGenerator = RandomOrdering.PermuteExterior(random),
                    InteriorMode = InteriorMode.Multiple,
                    BeforeSpreadAction = () => costs.SetCache(),
                };
                Refinement.SetFlowByDeterminant(domain.FlowRateDensity, state);
                state.InteriorFilter += Refinement.KeepClosestToConnection(lattice);

                var frozenFlow = this.FrozenFactor * domain.FlowRateDensity * lattice.Determinant;
                Func<INode, bool> frozenPredicate = frozenFlow != 0.0 ? n => n.Flow() < frozenFlow : null;
                // Overwrite with frozen set if specified
                if (frozen is not null)
                {
                    if (this.FreezeBranches)
                    {
                        frozenPredicate = n => !frozen.Contains(n.Parent.Branch.End.Position);
                    }
                    else
                    {
                        frozenPredicate = n => !frozen.Contains(n.Position);
                    }
                    // Frozen set must be updated on bifurcation?
                    if (this.UpdateFrozen)
                    {
                        state.TerminalPairBuildAction += (T, t) => frozen.Add(T.Upstream.Start.Position);
                    }
                }

                if (this.TerminalDeviation > 0)
                {
                    if (this.TerminalsMobile)
                    {
                        state.TerminalConstructor = (x, Q) => new MobileTerminal(x, Q, this.TerminalDeviation * stride);
                        state.OnEntry += () =>
                        {
                            var reinit = false;
                            foreach (var t in network.Terminals)
                            {
                                if (t is MobileTerminal mt)
                                {
                                    var x = mt.Position;
                                    mt.SetPosition(lattice.ToSpace(lattice.ClosestVectorBasis(x)));
                                    mt.PinningRadius = this.TerminalDeviation * stride;
                                    mt.Position = x;
                                }
                                else
                                {
                                    var xc = lattice.ToSpace(lattice.ClosestVectorBasis(t.Position));
                                    t.ReplaceWith(new MobileTerminal(xc, t.Flow, this.TerminalDeviation * stride)
                                    {
                                        Position = t.Position
                                    });
                                    reinit = true;
                                }
                            }
                            if (reinit)
                            {
                                state.Initialize();
                            }
                        };
                    }
                    else
                    {
                        var cr = new CylindricalRandom(random, this.TerminalDeviation * stride, 0);
                        state.TerminalConstructor = (x, Q) => new Terminal(x + cr.NextVector3(), Q);
                    }
                }

                // If any nodes are frozen, don't remove transients in case they are frozen
                state.AfterSpreadAction += () =>
                    optimizer.Optimize(network, costs,
                        this.SpacingMax * 0.1, this.SpacingMin * 0.1,
                        0.0, 0.0, t => state.Remove(t, false),
                        frozenPredicate, frozenPredicate is not null);
                state.AfterSpreadAction += () => optimizer.Trim(network, t => state.Remove(t, true));
                if (this.ClearMacula)
                {
                    state.AfterSpreadAction += () => Utility.ClearMacula(network, maculaRegion, false, t => state.Remove(t, false));
                }

                var singleBuild = new SingleBuild(state);
                state.TerminalPairPredicate = singleBuild.Predicate;
                state.BeforeSpreadAction += singleBuild.BeforeSpread;
                state.AfterSpreadAction += singleBuild.AfterSpread;
                state.TerminalPairBuildAction += singleBuild.OnBuild;

                if (stride >= this.TopologyOptimizationLimit)
                {
                    state.OnExit += () => optimizer.Optimize(network, costs,
                        this.SpacingMax * 0.1, this.SpacingMin * 0.1,
                        stride, domain.FlowRateDensity * lattice.Determinant, t => state.Remove(t, false),
                        frozenPredicate, frozenPredicate is not null);
                }
                if (frozenPredicate is null)
                {
                    state.OnExit += () => Utility.ResolveInternal(network, minPerturbation: this.SpacingMin * 0.01);
                }

                states.Add(state);
            }

            if (startingLeaves != null && !this.TerminalsMobile)
            {
                states[0].OnExit += () => Remove(network, startingLeaves);
            }

            var sequence = new LatticeSequence()
            {
                Elements = states
            };
            if (network.Root is null)
            {
                sequence.InitialTerminalCostFunction = Vector3.DistanceSquared;
                sequence.InitialTerminalPredicate = (S, T) => !S.Equals(T) && states[0].ExteriorPredicate(states[0].ClosestBasisFunction(T),T);               
                sequence.InitialTerminalOrderingGenerator = RandomOrdering.PermuteInitial(random);
                if (!sequence.Begin(5))
                {
                    throw new Exception("Could not initialize empty network");
                }
                network.Set(true, true, true);
            }
            sequence.Initialize();
            sequence.Complete();
        }

        private static void Remove(Network network, List<Terminal> terminals)
        {
            foreach (var t in terminals)
            {
                Topology.CullTerminal(t);
            }
            network.Set(true, true);
        }

        private void InitialCCO(Network network, Domain domain, Optimizer optimizer, Random random, out List<Terminal> initial)
        {
            initial = this.RemoveStartingLeaves && network.Root is not null ? network.Terminals.ToList() : null;
            if (this.PreSpacing == 0)
            {
                return;
            }

            var L = Utility.Lattice(this.PreSpacing, network);
            var dp = domain.DomainPredicate();
            var Z = LatticeActions.GetComponent(L, L.VoronoiCell.Connections,
                network.Source.Position - new Vector3(domain.MaculaRadius, 0, 0), x => dp(null, x));
            var Q = L.Determinant * domain.FlowRateDensity;
            var cr = new CylindricalRandom(random, this.TerminalDeviation * this.PreSpacing, 0);
            var TS = this.TerminalDeviation > 0 && !this.TerminalsMobile
                ? Z.Select(z => new Terminal(L.ToSpace(z) + cr.NextVector3(), Q))
                : Z.Select(z => new Terminal(L.ToSpace(z), Q));
            var T = new TerminalCollection(TS, new(), 1)
            {
                Random = random,
                Network = network
            };

            if (T.Waiting.Count > 0)
            {
                var sb = new SequentialBuilder();
                sb.All(network, T);

                if (this.RemoveStartingLeaves)
                {
                    Remove(network, initial);
                    initial = null;
                }
            }
            var costs = domain.Costs(network);
            optimizer.Optimize(network, costs,
                this.SpacingMax * 0.1, this.SpacingMin * 0.1,
                0.0, 0.0, null);
        }
    }
}
