import sys
import os

# Check for any already-cached nnunetv2 modules
to_remove = [key for key in sys.modules if key == "nnunetv2" or key.startswith("nnunetv2.")]
for key in to_remove:
    print(f'hyperresashs found existing nnunetv2 module {key} in sys.modules. Removing it to avoid conflicts with hyperresashs submodule nnunetv2.')
    del sys.modules[key]

# Insert the submodule path at the front so it takes priority
_submodule_path = os.path.join(os.path.dirname(__file__), "submodules", "nnUNet")
sys.path.insert(0, _submodule_path)


