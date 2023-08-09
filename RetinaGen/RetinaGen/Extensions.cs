using System;
using System.Collections.Generic;
using System.Text;
using Vascular.Geometry;
using Vascular.Structure;
using Vascular.Structure.Nodes;

namespace Retina
{
    static class Extensions
    {
        public static IEnumerable<string> Split(this string str, Predicate<char> pred)
        {
            var word = new StringBuilder();
            var index = 0;
            while (index < str.Length)
            {
                var ch = str[index];
                if (pred(ch))
                {
                    yield return word.ToString();
                    word.Clear();
                }
                else
                {
                    word.Append(ch);
                }
                ++index;
            }
            yield return word.ToString();
        }

        // Modified from Vascular/Structure/Extensions.cs in the Vascular.Networks project.
        public static void Transform(this Network network, Func<INode, Vector3> transform)
        {    
            foreach (var n in network.Nodes)
            {
                if (n is Terminal t)
                {
                    t.SetPosition(transform(t));
                }
                else if (n is Source s)
                {
                    s.SetPosition(transform(s));
                }

                // Test separately here as we may have a mobile terminal, and we need to move the
                // canonical position before we can move the actual location. No present support for
                // changing the pinning radius, but most transforms should be isometries anyway.
                if (n is IMobileNode m)
                {
                    m.Position = transform(m);
                }
            }
        }

        public static double FractionAlong(this Transient tr)
        {
            var l = 0.0;
            var b = tr.Parent.Branch;
            var s = b.Segments[0];
            while (s.Start != tr)
            {
                l += s.Length;
                s = s.End.Children[0];
            }
            return l / b.Length;
        }
    }
}
