import logging
from os import path

from training import RocketLearningSession

# --- Main Script ---
if __name__ == "__main__":
    # Define parameters for environment, agent, and training
    env_params = {
        'total_time_s': 2000,
        'step_size_s': 1.0e-2,
        'target_altitude_m': 75.0e+3,
        # 'start_altitude_m_lo': 60.4e+3,  # TRAINING
        # 'start_altitude_m_hi': 61.0e+3,  # TRAINING
        # 'start_altitude_m_hi': 61.5e+3,  # TRAINING
        'start_altitude_m_lo': 60.4e+3,  # VERIFICATION
        'start_altitude_m_hi': 61.0e+3,  # VERIFICATION
        'start_angle_deg_lo': 0.0,
        'start_angle_deg_hi': 0.0,
        'start_velocity_percent_orbital_lo': 0.0,
        'start_velocity_percent_orbital_hi': 0.0,
        'altitude_cutoff_lo': 60.0e+3,
        'altitude_cutoff_hi': 90.0e+3,
        # ... other environment parameters
    }
    agent_params = {
        'learning_rate': 1.0e-3,
        'gamma': 1.0 - 1.0e-2,
        'epsilon_lo': 0.01e+0,
        'epsilon_hi': 0.33e+0,
        'epsilon_decay': 1.0-1e-2,
        'flights_recorded': 0,
        'flight_steps_recorded': 100/env_params['step_size_s'],  # seconds * steps/s
        # TODO: needs improving to just use seconds
    }
    training_params = {
        'restart_episode_number': 0,
        'num_training_episodes': 20,
        'num_verification_episodes': 5,  # Add this for verification runs
        'epsilon_restart': 10,
        'flights_recorded': 10,
        'flight_seconds_replayed': 10,
        'large_epochs': 2,
        'small_epochs': 16,
        'scale': 1e+3,
        'folder': path.dirname(path.realpath(__file__)),
        'checkpoint_folder': '001',
        'load_checkpoint': False,
        # Will be done only for training runs
        'train_on_old_experience': True,
        'train_on_new_experience': True,
        'training_run': True,
        'verification_run': False,
        'crash_penalty': -1.0e+4
    }

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create the learning session
    session = RocketLearningSession(env_params, agent_params, training_params)

    # Trigger learning or verification
    if session.training_run:
        session.train(
            load=session._load_checkpoint,
            train_on_old_experience=session._train_on_old_experience,
            train_on_new_experience=session._train_on_new_experience,
        )
    if session.verification_run:
        session.verify(epsilon=0.0)  # Uncomment to run verification episodes
