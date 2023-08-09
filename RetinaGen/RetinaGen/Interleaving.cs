using System;
using Vascular.Intersections.Collision;
using Vascular.Structure;
using Vascular.Structure.Nodes;

namespace Retina
{
    class Interleaving
    {
        public double Radius { get; set; } = 5.0;
        public double PaddingFactor { get; set; } = 1.25;
        public double BifurcatingFlow { get; set; } = double.PositiveInfinity;
        public double FrozenFlow { get; set; } = double.PositiveInfinity;
        public int Iterations { get; set; } = 5;
        public bool Macula { get; set; } = false;

        public void Act(Network arterial, Network venous, Domain domain, Optimizer optimizer, Random random)
        {
            var nets = new[] { arterial, venous };           
            Resolve(nets, null);
        }
        
        private void Resolve(Network[] networks, Action<Terminal> onCull)
        {
            var rcrit = this.Radius * this.PaddingFactor;
            var resolver = new CollisionEnforcer(networks)
            {
                Recorder = new()
                {
                    ImmediateCull = (t, s) => t.Flow * 64 < s.Flow,
                    ImmediateCullDownstream = (a, b) => a.Radius <= rcrit && b.Radius <= rcrit && a.Radius <= b.Radius,
                },
                OperatingMode = CollisionEnforcer.Mode.External,
                ChangeGeometry = false,
                RadiusModification = b => b.Radius * this.PaddingFactor,
                TerminalCulled = onCull
            };
            resolver.Advance(1).Wait();
        }
    }
}
