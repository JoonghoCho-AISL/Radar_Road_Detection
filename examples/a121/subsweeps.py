# Copyright (c) Acconeer AB, 2022
# All rights reserved

import acconeer.exptool as et
from acconeer.exptool import a121


args = a121.ExampleArgumentParser().parse_args()
et.utils.config_logging(args)

client = a121.Client(**a121.get_client_args(args))
client.connect()
#start_distance = start_point * 2.5mm
start_distance = 100
start_point = start_distance / 2.5
#end_distance = start_point * 2.5mm + num_points * 2.5mm
end_distance = 200
num_points = (end_distance - start_point * 2.5) / 2.5

sensor_config = a121.SensorConfig(
    subsweeps=[
        a121.SubsweepConfig(
            
            start_point= 40,
            step_length=1,
            num_points=30,
            profile=a121.Profile.PROFILE_1,
            hwaas=10,
        ),
        # a121.SubsweepConfig(
        #     start_point=75,
        #     step_length=4,
        #     num_points=25,
        #     profile=a121.Profile.PROFILE_3,
        #     hwaas=20,
        # ),
    ],
    sweeps_per_frame=10,
)

client.setup_session(sensor_config)
client.start_session()

for i in range(3):
    result = client.get_next()

    print(f"\nResult {i + 1} subframes:")
    print(result.subframes)

client.stop_session()
client.disconnect()
