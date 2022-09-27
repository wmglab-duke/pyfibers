from src import *
import time


def main():
    start_time = time.time()  # Starting time of simulation

    potentials_path = 'example_inputs/potentials.dat'
    waveform_path = 'example_inputs/waveform.dat'
    stimulation = Stimulation()
    stimulation\
        .load_potentials(potentials_path)\
        .load_waveform(waveform_path)
    n_fiber_coords = len(stimulation.potentials)

    fiber = Fiber(diameter=8.7, fiber_mode='MRG_DISCRETE', temperature=37)
    fiber.generate(n_fiber_coords)

    stimulation.apply_intracellular(fiber,
                                    delay=0,
                                    pw=0,
                                    dur=0,
                                    freq=0,
                                    amp=0,
                                    ind=0,
                                    )

    saving = Saving('.',
                    stimulation.dt,
                    fiber,
                    space_vm=True,
                    space_times=[1, 9, 20],
                    time_gating=True,
                    locs=[0.1, 0.3, 0.9],
                    runtime=True)

    recording = Recording(fiber)

    fiber.submit(stimulation,
                 saving,
                 recording,
                 start_time,
                 protocol_mode='ACTIVATION_THRESHOLD',
                 t_init_ss=-200,
                 dt_init_ss=10,
                 )


if __name__ == '__main__':
    main()
