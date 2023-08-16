using Retina;
using System.IO;
using System.Text.Json;

// Overview
// Prerequisites (Python)
// - Initial networks to be created by L-system (in plane), written to Amira file
// - Amira file converted to JSON
// - JSON file created containing data for "Discs" class
// This program
// - Paramters loaded from JSON file describing the "Discs" class
//   - This includes the paths to the AmiraJSON files
// - Network grown using lattice sequence approach to provide basic structure
// - Macula cleared and regrown to generate radially penetrating vessels
// - Networks projected to quadratic if desired, combined and collisions resolved
// - Networks written to separate CSV files in (x y z X Y Z r) form for segments

if (args.Length == 0)
{
    IO.WriteDefaults();
    return 0;
}
var io = JsonSerializer.Deserialize<IO>(File.ReadAllText(args[0]), IO.JsonSerializerOptions);
var nets = io.Read();

Actions.Act(nets, io);

io.Write(nets);
return 0;