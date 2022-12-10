"""Parameters for creating fiber models."""
from src.wmglab_neuron import FiberModel

FiberTypeParameters = {  # TODO: this needs comments
    FiberModel.MRG_DISCRETE: {
        "myelinated": True,
        "node_length": 1.0,
        "paranodal_length_1": 3.0,
        "diameters": [1.0, 2.0, 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0],
        "delta_zs": [100, 200, 500, 750, 1000, 1150, 1250, 1350, 1400, 1450, 1500],
        "paranodal_length_2s": [5, 10, 35, 38, 40, 46, 50, 54, 56, 58, 60],
        "gs": [None, None, 0.605, 0.63, 0.661, 0.69, 0.7, 0.719, 0.739, 0.767, 0.791],
        "axonDs": [0.8, 1.6, 3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7],
        "nodeDs": [0.7, 1.4, 1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5],
        "nls": [15, 30, 80, 100, 110, 120, 130, 135, 140, 145, 150],
        "v_rest": -80,  # millivolts
    },
    FiberModel.MRG_INTERPOLATION: {
        "myelinated": True,
        "node_length": 1.0,
        "paranodal_length_1": 3.0,
        "paranodal_length_2": lambda d: -0.1652 * d**2 + 6.354 * d - 0.2862,
        "delta_z": lambda d: -8.215 * d**2 + 272.4 * d - 780.2 if d >= 5.643 else 81.08 * d + 37.84,
        "nl": lambda d: -0.4749 * d**2 + 16.85 * d - 0.7648,
        "nodeD": lambda d: 0.01093 * d**2 + 0.1008 * d + 1.099,
        "axonD": lambda d: 0.02361 * d**2 + 0.3673 * d + 0.7122,
        "v_rest": -80,  # millivolts
    },
    FiberModel.SUNDT: {
        "myelinated": False,
        "delta_zs": 8.333,
        "v_rest": -60,  # millivolts
    },
    FiberModel.MRG_DISCRETE.TIGERHOLM: {
        "myelinated": False,
        "delta_zs": 8.333,
        "v_rest": -55,  # millivolts
    },
    FiberModel.RATTAY: {
        "myelinated": False,
        "delta_zs": 8.333,
        "v_rest": -70,  # millivolts
    },
    FiberModel.SCHILD97: {
        "myelinated": False,
        "delta_zs": 8.333,
        "v_rest": -48,  # millivolts
    },
    FiberModel.SCHILD94: {
        "myelinated": False,
        "delta_zs": 8.333,
        "v_rest": -48,  # millivolts
    },
}
