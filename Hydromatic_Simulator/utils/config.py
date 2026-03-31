import numpy as np

config = {
    'num-coord': 16,
    'init-pos': np.stack([
        np.linspace(4.0625, 65.0, 16),
        np.zeros(16)
    ], axis=-1),
    'nodal-dist': 4.0625,

    'decimal-pt': 4,

    'timesteps': ['30 min', '60 min', '120 min', '180 min', '240 min',
                  '300 min', '420 min', '600 min', '1440 min (Equilibrium)'],
    'num-timesteps': 9,
    
    'structure-dim': 65,
    'nodal-dim': 2,
    'feature-dim': 281,
    'lstm-dim': 367,
    'dense-dim': 385,
    'batch-size': 208,
}
