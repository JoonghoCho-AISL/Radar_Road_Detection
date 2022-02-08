from acconeer.exptool.a111.algo import ModuleFamily, ModuleInfo

from .plotting import PGUpdater
from .processing import ButtonPressProcessor, get_processing_config, get_sensor_config


module_info = ModuleInfo(
    key="button_press_sparse",
    label="Button Press (sparse)",
    pg_updater=PGUpdater,
    processing_config_class=get_processing_config,
    module_family=ModuleFamily.EXAMPLE,
    sensor_config_class=get_sensor_config,
    processor=ButtonPressProcessor,
    multi_sensor=False,
    docs_url=None,
)
