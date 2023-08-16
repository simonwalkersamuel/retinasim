using System.Collections.Generic;
using System.Linq;
using Vascular;
using Vascular.Intersections.Collision;
using Vascular.Structure;

namespace Retina
{
    delegate void Step(Network[] networks, IO io);

    static class Actions
    {
        private static readonly Dictionary<string, Step> steps = new()
        {
            { CoarseName, CoarseStep },
            { InterleavingName, InterleavingStep },
            { MaculaName, MaculaStep },
            { ResolveName, ResolveStep },
            { OptimizeName, OptimizeStep }
        };

        public static void Act(Network[] networks, IO io)
        {
            var actions = io.Actions
                .Split(ch => !char.IsLetter(ch))
                .Where(str => !string.IsNullOrWhiteSpace(str))
                .Select(str => str.Trim().ToLowerInvariant());
            foreach (var action in actions)
            {
                if (steps.TryGetValue(action, out var step))
                {
                    step(networks, io);
                }
            }
        }

        public static string DefaultActions => string.Join(';', CoarseName, InterleavingName, MaculaName, ResolveName);

        private static string CoarseName => "coarse";
        private static void CoarseStep(Network[] networks, IO io)
        {
            networks.RunAsync(n =>
            {
                io.Major.Grow(n, io.Domain, io.Optimizer, io.Random());
            }).Wait();
        }

        private static string InterleavingName => "interleaving";
        private static void InterleavingStep(Network[] networks, IO io)
        {
            io.Interleaving.Act(networks[0], networks[1], io.Domain, io.Optimizer, io.Random());
        }

        private static string MaculaName => "macula";
        private static void MaculaStep(Network[] networks, IO io)
        {
            networks.RunAsync(n =>
            {
                io.Macula.Grow(n, io.Domain, io.Optimizer, io.Random());
            }).Wait();
        }

        private static string ResolveName => "resolve";
        private static void ResolveStep(Network[] networks, IO io)
        {
            var enforcer = new CollisionEnforcer(networks)
            {
                Recorder = new()
                {
                    MinimumNodePerturbation = io.Major.SpacingMin / 100.0,
                    ImmediateCull = (t, s) => t.Flow * 64 < s.Flow,
                },
                OperatingMode = CollisionEnforcer.Mode.All,
                InternalTestStagger = 4,
                RadiusModification = b => b.Radius * 1.25,
            };
            enforcer.Resolve().Wait();
        }

        private static string OptimizeName => "optimize";
        private static void OptimizeStep(Network[] networks, IO io)
        {
            networks.RunAsync(n => io.Optimizer.Standalone(n, io)).Wait();
        }
    }
}
