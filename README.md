# aind-nwb-utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## Usage
This package is intended to simplify and standardize the process of creating and interacting with NWB files. It is still in the early development stages, and the API and naming conventions are subject to change at any time.

## Combining NWB files with `NWBCombineIO`

`NWBCombineIO` merges one or more *sub* NWB files into a *main* NWB file and, optionally, exports the result to disk. Data is merged in memory — the input files are opened read-only and are never modified.

The main file is the one that carries device information or units data; files carrying trials data are typically passed as sub files. This split is a current limitation.

### Basic usage

As a context manager, which yields the merged `NWBFile` and the main IO handle:

```python
from pathlib import Path
from aind_nwb_utils import NWBCombineIO

main_path = Path("/data/ecephys.nwb")
sub_paths = [Path("/data/behavior.nwb")]

with NWBCombineIO(main_path, sub_paths) as (nwb_file, main_io):
    print(nwb_file.trials.to_dataframe())
```

As a standalone object, when you need the merged file to outlive a `with` block:

```python
combiner = NWBCombineIO(main_path, sub_paths)
nwb_file = combiner.read()
# ... work with nwb_file ...
combiner.close()
```

The IO handles stay open until `close()` (or context-manager exit) so that lazily loaded datasets remain readable. Accessing data from a merged file after closing will fail.

### Writing the combined file

```python
with NWBCombineIO(main_path, sub_paths) as (nwb_file, main_io):
    ...

combiner = NWBCombineIO(main_path, sub_paths)
combiner.write(Path("/results/combined.nwb.zarr"), format="zarr")
combiner.close()
```

`format` accepts `"zarr"` (default) or `"hdf5"`; anything else raises `ValueError`. `write()` opens the inputs itself if `read()` has not already been called. The export uses `link_data=False`, so datasets are copied into the output rather than linked back to the source files — the output stands on its own.

### Input formats

Input paths may be HDF5 files or Zarr stores, in any combination; the correct IO class is chosen per file by `determine_io`, which checks in order:

1. **File structure** — the HDF5 magic signature, or a Zarr root key (`.zgroup`, `.zmetadata`, `zarr.json`). This is the strongest signal because it ignores naming.
2. **Extension** — `.nwb` means HDF5, `.zarr` means Zarr.
3. **Fallback** — a directory is treated as a Zarr store, anything else as HDF5.

`s3://` paths are supported. Structure detection over S3 requires `fsspec`/`s3fs`; if those are unavailable or the probe fails, detection falls back to the extension and any genuine access problem surfaces at read time.

### Extension namespaces

Each NWB file caches the namespaces it was written with, so reading files independently would give each IO handle a TypeMap that only knows that one file's extensions. Because the export runs through the main file's build manager, a container from a sub file whose extension the main file does not know would be downgraded to the nearest base type it does know — an `ndx-hed` `HedLabMetaData` would silently become a plain `LabMetaData`, dropping every extension-specific field.

`NWBCombineIO` avoids this by loading the cached namespaces of *all* inputs into a single `TypeMap` and sharing it across every handle (each handle still gets its own `BuildManager`, since those cache builder/container pairs per file). Extension types, their attributes, and their cached specs survive both the merge and the export. A file with no cached specs logs a warning and is still read using the namespaces already loaded.

### Merge rules

Each field of each sub file is merged into the main file according to its type:

| Field type | Behavior |
| --- | --- |
| Scalars and metadata (`str`, `datetime`, `list`, `Subject`, tuples, numpy/zarr arrays) | The main file's value wins. If the main file has no value, the sub file's is copied in, so metadata present in only one input (`institution`, `experimenter`, `was_generated_by`, `subject`, ...) still reaches the output. A mismatch between two present values is logged as a warning. |
| `TimeIntervals` (`intervals`, `trials`, `epochs`, `invalid_times`) | Added under `/intervals/<name>`, since all four NWBFile fields live there. |
| `EventsTable` | If a table of the same name already exists, columns are merged into it — new columns are added, and columns present in both have the sub file's data appended. Otherwise the whole table is added. |
| Dict-like fields (`acquisition`, `processing`, `devices`, `stimulus`, ...) | Each entry is added by name through the corresponding public `NWBFile` adder (`add_acquisition`, `add_device`, ...). |
| Anything else | Raises `TypeError`. |

Two behaviors are worth knowing about:

- **Name collisions favor the main file.** If a sub file contains an object whose name already exists in the same field of the main file, the sub file's copy is skipped and the event is logged. Merging is not order-independent when names overlap.
- **64-bit data is downcast.** Plain `TimeSeries` and `VectorData` holding `float64`/`int64` are rebuilt as `float32`/`int32` to avoid dtype conflicts between files. `TimeSeries` *subclasses* are passed through untouched, because rebuilding them as a plain `TimeSeries` would discard their neurodata type.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
