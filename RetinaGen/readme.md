# RetinaGen
RetinaGen is an executable built on the [Vascular.Networks](https://github.com/AndrewAGuy/vascular-networks) package, intended to take existing retina-like networks and further grow them or start from nothing in a retina-like domain. It includes a few features (particularly around the macula) based on conversations with opthalmologists and operates under the assumption that retinal vasculature (which sits in front of the receptors) is optimal when it minimizes the obstruction of light to these cells.

## Building and using RetinaGen as part of RetinaSim
1) Install the .NET SDK, version 6.0 or greater (RetinaGen targets the .NET 6.0 runtime).
2) Go to `RetinaGen/RetinaGen` and build the executable. This will automatically collect Vascular.Networks and build both Vascular.IO.Amira and RetinaGen.
3) Point RetinaSim to the executable and output directories via `config.py`.

If you're not experienced with .NET: this should work (starting from the RetinaSim base directory):
```bash
$ sudo apt-get update && sudo apt-get install dotnet-sdk-6.0
$ cd RetinaGen/RetinaGen
$ dotnet build -c Release
```
By default, output files will be found in `data/RetinaGen`.

### Directory structure
It is intended that _Vascular.IO.Amira_ will be moved to a new home and published as a separate package at some point. When that happens, everything will probably be moved up one level in the directory structure and the default paths here will be updated.

## Standalone usage
RetinaGen is intended to be used as part of the RetinaSim pipeline and as such has a somewhat awkward interface. It expects its parameters to be provided as a JSON file following the structure of the `IO` class, given as the only command line argument, which it then deserializes and acts on. Run RetinaGen without arguments to generate an example file with the default values to `example.json`.
RetinaGen can be used on a single network for growth/optimization steps but requires two for collision resolution steps.

To use RetinaGen as a standalone executable, you'll need to provide the networks in the format expected by _Vascular.IO.Amira_ (you can convert an Amira HxSpatialGraph with suitably named data fields into this format by using the `amirajson.py` module from [pymira](https://github.com/CABI-SWS/pymira)) or as a CSV file of segment definitions [x y z X Y Z r]. It assumes the largest vessel radius is the root and keeps this fixed.

## Licensing
This sub-repository is licensed under the GNU Affero General Public License, Version 3 (AGPL-3.0).