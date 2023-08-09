using System;
using System.Collections.Generic;
using Vascular.Structure;
using Vascular.Geometry.Lattices;
using Vascular.Geometry.Lattices.Transformed;
using Vascular.Geometry;
using Vascular.Structure.Nodes;
using Vascular.Intersections.Collision;
using Vascular.Intersections.Segmental;
using Vascular;
using Vascular.Geometry.Lattices.Manipulation;

namespace Retina
{
    static class Utility
    {
        public static List<double> Spacings(double min, double max, int levels)
        {
            var spacings = new List<double>(levels + 1)
            {
                max
            };
            if (levels > 0)
            {
                var logK = (Math.Log(min) - Math.Log(max)) / levels;
                var k = Math.Exp(logK);
                for (var i = 1; i < levels; ++i)
                {
                    spacings.Add(max * Math.Pow(k, i));
                }
                spacings.Add(min);
            }
            return spacings;
        }

        public static Lattice Lattice(double stride, Network network, bool hexagonal = true)
        {
            Lattice baseLattice = hexagonal 
                ? new HexagonalPrismLattice(stride, 1, HexagonalPrismLattice.Connection.Triangle)
                : new CuboidLattice(stride, 1, stride, CuboidLattice.Connection.SquareEdges);

            if (network.Output)
            {
                var offset = stride * 0.5;
                return new OffsetLattice(baseLattice, new Vector3(offset, offset, 0));
            }
            return baseLattice;
        }

        public static void ResolveInternal(Network network, Action<Terminal> onCull = null,
            double minPerturbation = 0.0, double Qcrit = 32)
        {
            network.Root.SetLogical();
            network.Source.CalculatePhysical();

            var resolver = new CollisionEnforcer(network.AsArray())
            {
                Recorder = new()
                {
                    RecordTopology = true,
                    MinimumNodePerturbation = minPerturbation,
                    // Prevents wavy patterns forming in major vessels
                    ImmediateCull = (t, s) => t.Flow * Qcrit < s.Flow,
                },
                TerminalCulled = onCull,
                OperatingMode = CollisionEnforcer.Mode.Internal
            };

            resolver.Resolve().Wait();
        }

        public static void ClearMacula(Network network, SegmentRegion macula,
            bool tryResolve = false, Action<Terminal> onCull = null)
        {
            var enforcer = new SegmentEnforcer(network.AsArray(), macula.AsArray())
            {
                Recorder = new()
                {
                    TotalConsumptionRatio = 10
                },
                Penalizer = new()
                {
                    Decay = 0,
                    Penalty = 1,
                    Threshold = tryResolve ? 10 : 1
                },
                ChangeGeometry = tryResolve,
                TerminalCulled = onCull
            };

            enforcer.Resolve().Wait();
        }

        public static void Resolve(Network[] networks, 
            Lattice matchingLattice = null, double minPerturbation = 0.0)
        {
            LatticeActions.MatchTerminals(matchingLattice, networks);
            var enforcer = new CollisionEnforcer(networks)
            {
                Recorder = new()
                {
                    MinimumNodePerturbation = minPerturbation,
                    ImmediateCull = (t, s) => t.Flow * 64 < s.Flow,
                },
                OperatingMode = CollisionEnforcer.Mode.All,
                InternalTestStagger = 4,
                RadiusModification = b => b.Radius * 1.25,
            };
            enforcer.Resolve().Wait();
        }
    }
}
