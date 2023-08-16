# RetinaGen
RetinaGen is an executable built on the [Vascular.Networks](https://github.com/AndrewAGuy/vascular-networks) package, intended to take existing retina-like networks and further grow them or start from nothing in a retina-like domain. It includes a few features (particularly around the macula) based on conversations with opthalmologists and operates under the assumption that retinal vasculature (which sits in front of the receptors) is optimal when it minimizes the obstruction of light to these cells.

## Building
In the `/RetinaGen` directory, execute `dotnet build` with whatever options you want. This should collect _Vascular.Networks_ and build the dependency used for loading AmiraMesh files which have been converted using the `amirajson.py` utility in [pymira](https://github.com/CABI-SWS/pymira), _Vascular.IO.Amira_. Everything should then be output in `/RetinaGen/bin/[Debug/Release]/net6.0/` - copy everything from there if you need to move it as it doesn't condense everything required into a single DLL.

It is intended that _Vascular.IO.Amira_ will be moved to a new home and published as a separate package at some point. When that happens, everything will probably be moved up one level in the directory structure.

## Licensing
This sub-repository is licensed under the GNU Affero General Public License, Version 3 (AGPL-3.0).