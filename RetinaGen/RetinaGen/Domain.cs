using System;
using Vascular;
using Vascular.Construction.LSV;
using Vascular.Geometry;
using Vascular.Intersections.Segmental;
using Vascular.Optimization;
using Vascular.Structure;

namespace Retina
{
    class Domain
    {
        public double LengthUnitsPerMetre { get; set; } = 1e6;

        public double[] OpticDiscPosition { get; set; } = new double[3] { 0, 0, 0 };
        public double[] FoveaPosition { get; set; } = new double[3] { 4.76e3, 0, 0 };
        public double MaculaRadius { get; set; } = 2.75e3;
        public double FoveaRadius { get; set; } = 0.75e3;

        public double DomainRadius { get; set; } = 12.5e3;
        public double[] DomainCentre { get; set; } = new double[3] { 4.76e3, 0, 0 };
        public int SemiCircle { get; set; } = 1;

        public double Viscosity { get; set; } = 4e-3; // Pa s
        public double FlowRateDensity { get; set; } = 1.0 / 60.0; // In L^3/s / L^3 for length units L

        public double ArteryMurrayExponent { get; set; } = 3.0;
        public double VeinMurrayExponent { get; set; } = 3.0;

        public double ArteryRootRadius { get; set; } = 75;
        public double VeinRootRadius { get; set; } = 100;

        public Func<Vector3, bool> MaculaPredicate()
        {
            var fovea = Vector3.FromArrayPermissive(this.FoveaPosition);
            var r2 = Math.Pow(this.MaculaRadius, 2);
            return x => Vector3.DistanceSquared(x, fovea) <= r2;
        }

        public Func<Vector3, bool> MaculaFoveaPredicate()
        {
            var fovea = Vector3.FromArrayPermissive(this.FoveaPosition);
            var r2m = Math.Pow(this.MaculaRadius, 2);
            var r2f = Math.Pow(this.FoveaRadius, 2);
            return x =>
            {
                var d2 = Vector3.DistanceSquared(x, fovea);
                return d2 <= r2m && d2 >= r2f;
            };
        }

        public Func<Vector3, bool> FoveaDistancePredicate(double distance)
        {
            var fovea = Vector3.FromArrayPermissive(this.FoveaPosition);
            var r2 = Math.Pow(distance, 2);
            return x => Vector3.DistanceSquared(x, fovea) <= r2;
        }

        public ExteriorPredicate DomainPredicate()
        {
            var r2 = Math.Pow(this.DomainRadius, 2);
            var rc = Vector3.FromArrayPermissive(this.DomainCentre);
            return this.SemiCircle switch
            {
                > 0 => (z, x) => x.y >= 0 && Vector3.DistanceSquared(rc, x) <= r2,
                < 0 => (z, x) => x.y <= 0 && Vector3.DistanceSquared(rc, x) <= r2,
                0 => (z, x) => Vector3.DistanceSquared(rc, x) <= r2
            };
        }

        public SegmentRegion MaculaRegion()
        {
            var foveaPosition = Vector3.FromArrayPermissive(this.FoveaPosition);
            var maculaSegment = Segment.MakeDummy(
                foveaPosition - new Vector3(0, 0, this.MaculaRadius),
                foveaPosition + new Vector3(0, 0, this.MaculaRadius),
                this.MaculaRadius);
            maculaSegment.GenerateBounds();
            return new SegmentList(maculaSegment.AsArray());
        }

        public HierarchicalCosts Costs(Network n)
        {
            // For now, only consider area.
            return new HierarchicalCosts(n)
            {
                WorkFactor = 0,
            }.AddSchreinerCost(1, 1, 1);
        }
    }
}
