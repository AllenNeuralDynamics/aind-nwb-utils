"""Regenerate ``merge_test_extension.nwb``.

The fixture holds a ``LabMetaData`` subclass from a custom namespace
(``ndx-merge-test``) so the merge tests can prove that extension types are
not silently downgraded to their base type.

The namespace is deliberately *not* one the test suite registers. The test
process only ever sees it through the namespaces cached inside this file,
which is exactly the path :meth:`NWBCombineIO._build_type_map` exists to
support. Registering the extension inside the test process instead would
put it in pynwb's global type map and mask the regression entirely.

Usage::

    python tests/resources/generate_merge_test_extension.py
"""

import datetime
import tempfile
from pathlib import Path

from hdmf.utils import docval, get_docval, popargs
from pynwb import NWBHDF5IO, NWBFile, load_namespaces, register_class
from pynwb.file import LabMetaData
from pynwb.spec import NWBAttributeSpec, NWBGroupSpec, NWBNamespaceBuilder

NAMESPACE = "ndx-merge-test"
FIXTURE_PATH = Path(__file__).parent / "merge_test_extension.nwb"


def build_namespace(spec_dir: Path) -> Path:
    """Write the ``ndx-merge-test`` namespace to ``spec_dir``.

    Parameters
    ----------
    spec_dir : Path
        Directory to write the namespace and extension YAML into.

    Returns
    -------
    Path
        Path to the namespace YAML file.
    """
    spec = NWBGroupSpec(
        doc="lab metadata carrying an extension-only attribute",
        neurodata_type_def="MergeTestLabMetaData",
        neurodata_type_inc="LabMetaData",
        attributes=[
            NWBAttributeSpec(
                name="tag", doc="an extension-only attribute", dtype="text"
            )
        ],
    )
    builder = NWBNamespaceBuilder(
        doc="fixture namespace for aind-nwb-utils merge tests",
        name=NAMESPACE,
        version="0.1.0",
        author="Allen Institute for Neural Dynamics",
        contact="aind",
    )
    builder.include_type("LabMetaData", namespace="core")
    builder.add_spec(f"{NAMESPACE}.extensions.yaml", spec)
    builder.export(f"{NAMESPACE}.namespace.yaml", outdir=str(spec_dir))
    return spec_dir / f"{NAMESPACE}.namespace.yaml"


def main() -> None:
    """Build the namespace and write the fixture to disk."""
    spec_dir = Path(tempfile.mkdtemp())
    load_namespaces(str(build_namespace(spec_dir)))

    @register_class("MergeTestLabMetaData", NAMESPACE)
    class MergeTestLabMetaData(LabMetaData):
        """LabMetaData subclass with one extension-only attribute."""

        __nwbfields__ = ("tag",)

        @docval(
            *get_docval(LabMetaData.__init__),
            {
                "name": "tag",
                "type": str,
                "doc": "the extension-only attribute",
            },
        )
        def __init__(self, **kwargs):
            """Initialize with the extension-only ``tag`` attribute."""
            tag = popargs("tag", kwargs)
            super().__init__(**kwargs)
            self.tag = tag

    nwb_file = NWBFile(
        session_description="extension fixture for merge tests",
        identifier="merge-test-extension",
        session_start_time=datetime.datetime(
            2024, 1, 1, tzinfo=datetime.timezone.utc
        ),
    )
    nwb_file.add_lab_meta_data(
        MergeTestLabMetaData(name="merge_test_meta", tag="hello")
    )

    with NWBHDF5IO(str(FIXTURE_PATH), "w") as io:
        io.write(nwb_file)

    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
