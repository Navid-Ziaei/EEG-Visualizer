from eeg_visualizer.eeg_visualizer import *
from eeg_visualizer.import_data import *

FIGURE_HEIGHT = '720px'

# Load sample data

base_path = 'F:/Datasets/CLEAR Data/Processed/'
settings = Settings(target_class='color', task='flicker')
settings.patient = ['p05']
settings.features = ['raw']

# Create path and load data
paths = Paths(base_path, settings)
paths.create_paths()
data_loader = DataLoader(paths, settings)
dataset = data_loader.load_raw_data()

column_list = ['TrialNum', 'ColorLev', 'fixation_time', 'display_time']
df = dataset.trial_info[column_list]

df_channels = pd.DataFrame({'channel name': dataset.channel_name})
df_channels['index_col'] = df_channels.index

visualizer = VizServer(dataset=dataset, trial_info=df, channels_info=df_channels)
visualizer.run_server()


