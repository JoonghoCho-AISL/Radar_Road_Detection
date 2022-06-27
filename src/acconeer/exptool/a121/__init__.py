import sys

from ._cli import ExampleArgumentParser, get_client_args
from ._core import (
    _H5PY_STR_DTYPE,
    PRF,
    Client,
    ClientError,
    ClientInfo,
    H5Record,
    H5Recorder,
    IdleState,
    InMemoryRecord,
    Metadata,
    PersistentRecord,
    Profile,
    Record,
    Recorder,
    Result,
    SensorConfig,
    SensorInfo,
    ServerInfo,
    SessionConfig,
    StackedResults,
    SubsweepConfig,
    ValidationError,
    ValidationResult,
    ValidationWarning,
    load_record,
    open_record,
    save_record,
    save_record_to_h5,
)
from ._core_ext import _ReplayingClient, _StopReplay
from ._perf_calc import _PerformanceCalc


if "pytest" not in sys.modules:
    import warnings

    warnings.warn(
        "The a121 package is currently an unstable API and may change at any time.",
        FutureWarning,
    )
