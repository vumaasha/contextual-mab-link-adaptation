from gym.envs.registration import register

# CQI Reporting interval 1 TTI, No interference
register( id = 'MCSSelection-v0',
          entry_point = 'gym_radio_link.envs:MCSSelectionEnv',
          kwargs = { 'ue_relative_speed': -1,
                     'cqi_reporting_interval': 1,
                     'intercell_interference': False } )

# CQI Reporting interval 1 TTI, Intercell interference
register( id = 'MCSSelection-v1',
          entry_point = 'gym_radio_link.envs:MCSSelectionEnv',
          kwargs = { 'ue_relative_speed': -1,
                     'cqi_reporting_interval': 1,
                     'intercell_interference': True } )

# CQI Reporting interval 10 TTIs, No Intercell interference
register( id = 'MCSSelection-v2',
          entry_point = 'gym_radio_link.envs:MCSSelectionEnv',
          kwargs = { 'ue_relative_speed': -1,
                     'cqi_reporting_interval': 10,
                     'intercell_interference': False } )

# CQI Reporting interval 10 TTIs, Intercell interference
register(id = 'MCSSelection-v3',
         entry_point = 'gym_radio_link.envs:MCSSelectionEnv',
         kwargs={'ue_relative_speed': -1,
                 'cqi_reporting_interval': 10,
                 'intercell_interference': True})
