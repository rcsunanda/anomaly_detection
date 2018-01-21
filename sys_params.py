
import anomaly_detection.data_synthesizer as gen

class SystemParameters:
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    def print(self):
        members_dict = dict(self)
        for key, val in members_dict.items():
            print("\t{}={}".format(key, val))


def set_synthetic_params(sys_params):

    # Data synthesis
    sys_params.data_generation_func = gen.generate_high_dim_complex_series
    sys_params.train_data_t_range = (-1, 1)
    #sys_params.train_t_range = (-math.pi, math.pi) # For trig functions
    sys_params.test_data_t_range = (-2, 2)
    #sys_params.test_data_t_range = (-2*math.pi, 2*math.pi) # For trig functions

    # LSTM network architecture
    sys_params.dimension = 3
    sys_params.input_timesteps = 3
    sys_params.output_timesteps = 1
    sys_params.hidden_layer_units = 50

    # Training params
    sys_params.batch_size = 20  # Mini batch size in GD/ other algorithm
    sys_params.epcohs = 20  # 50 is good

    # Anomaly and detection
    sys_params.anomaly_rate = 0.05
    sys_params.diff_anomaly_threshold = 0.12

    # Visualization
    sys_params.max_dim_to_plot = sys_params.dimension


def set_real_dataset_1_params(sys_params):

    # Data files
    # sys_params.training_data_file = 'data/Matching_1_1_Sequencer_1_small_subset_training.csv'
    # sys_params.test_data_file = 'data/Matching_1_1_Sequencer_1_small_subset_test.csv'
    sys_params.training_data_file = 'data/Matching_1_1_Sequencer_1_full_training.csv'
    sys_params.test_data_file = 'data/Matching_1_1_Sequencer_1_full_test.csv'

    # LSTM network architecture
    sys_params.dimension = -1    # Set when loading data
    sys_params.input_timesteps = 3
    sys_params.output_timesteps = 1
    sys_params.hidden_layer_units = 50

    # Training params
    sys_params.batch_size = 20  # Mini batch size in GD/ other algorithm
    sys_params.epcohs = 5  # 50 is good, 20 is also enough

    # Anomaly and detection
    sys_params.anomaly_rate = 0.05
    sys_params.diff_anomaly_threshold = 0.12

    # Visualization
    sys_params.max_dim_to_plot = 3


def init_system_params(system_name):
    print("init_system_params: Initializing system parameters")

    sys_params = SystemParameters()
    sys_params.system_name = system_name

    if system_name == 'synthetic_dataset':
        set_synthetic_params(sys_params)
    elif system_name == 'real_dataset_1':
        set_real_dataset_1_params(sys_params)
    else:
        print("Unknown system_name: {}".format(system_name))
        assert False

    sys_params.print()

    return sys_params