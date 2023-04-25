import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
from pathlib import Path
import os
import shutil


class DataLoader:
    def __init__(self, paths, settings):
        self.target_class = settings.target_class
        self.patient = settings.patient
        self.task = 'FlickerShapes'  # 'FlickerShapes' or 'Flicker'
        self.paths = paths
        self.results = {}

    def load_data(self):

        data_matlab_class1_hgp = io.loadmat(self.paths.path_dataset_hgp[0])
        data_matlab_class2_hgp = io.loadmat(self.paths.path_dataset_hgp[1])
        data_matlab_class1_erp = io.loadmat(self.paths.path_dataset_erp[0])
        data_matlab_class2_erp = io.loadmat(self.paths.path_dataset_erp[1])

        if self.target_class == 'color':
            data_matlab_class1_hgp = data_matlab_class1_hgp['ft_FrerqData_Black_filtered'][0]
            data_matlab_class2_hgp = data_matlab_class2_hgp['ft_FrerqData_White_filtered'][0]
            data_matlab_class1_erp = data_matlab_class1_erp['ft_data_Black_lp_ds'][0]
            data_matlab_class2_erp = data_matlab_class2_erp['ft_data_White_lp_ds'][0]
        elif self.target_class == 'shape':
            data_matlab_class1_hgp = data_matlab_class1_hgp['ft_FrerqData_Shape1_filtered'][0]
            data_matlab_class2_hgp = data_matlab_class2_hgp['ft_FrerqData_Shape2_filtered'][0]
            data_matlab_class1_erp = data_matlab_class1_erp['ft_data_Shape1_lp_ds'][0]
            data_matlab_class2_erp = data_matlab_class2_erp['ft_data_Shape2_lp_ds'][0]
        elif self.target_class == 'tone':
            data_matlab_class1_hgp = data_matlab_class1_hgp['ft_FrerqData_Tone1_filtered'][0]
            data_matlab_class2_hgp = data_matlab_class2_hgp['ft_FrerqData_Tone2_filtered'][0]
            data_matlab_class1_erp = data_matlab_class1_erp['ft_data_Tone1_lp_ds'][0]
            data_matlab_class2_erp = data_matlab_class2_erp['ft_data_Tone2_lp_ds'][0]
        else:
            raise ValueError('target class ' + self.target_class + ' is not defined')

        time = np.squeeze(data_matlab_class1_hgp['time'][0][0][0])

        # specify data dimesions
        num_trials_class1 = data_matlab_class1_hgp['trial'][0][0].shape[0]
        num_trials_class2 = data_matlab_class2_hgp['trial'][0][0].shape[0]

        num_channels, num_samples_erp = data_matlab_class1_erp['trial'][0][0][0].shape
        num_channels, num_samples_hgp = data_matlab_class1_hgp['trial'][0][0][0].shape

        if num_samples_erp != num_samples_hgp:
            print('ERP number of samples is ' + str(num_samples_hgp) +
                  ' but HGP number of samples is ' + str(num_samples_hgp))
            num_samples = np.min([num_samples_hgp, num_samples_hgp])
        else:
            num_samples = num_samples_erp

        time = time[:num_samples]

        # change data into numpy array
        data_erp_class1 = np.zeros((num_trials_class1, num_channels, num_samples))
        data_hgp_class1 = np.zeros((num_trials_class1, num_channels, num_samples))
        for i in range(num_trials_class1):
            data_erp_class1[i, :, :] = data_matlab_class1_erp['trial'][0][0][i][:, :num_samples]
            data_hgp_class1[i, :, :] = data_matlab_class1_hgp['trial'][0][0][i][:, :num_samples]

        data_erp_class2 = np.zeros((num_trials_class2, num_channels, num_samples))
        data_hgp_class2 = np.zeros((num_trials_class2, num_channels, num_samples))
        for i in range(num_trials_class2):
            data_erp_class2[i, :, :] = data_matlab_class2_erp['trial'][0][0][i][:, :num_samples]
            data_hgp_class2[i, :, :] = data_matlab_class2_hgp['trial'][0][0][i][:, :num_samples]

        # channel names
        channel_names = data_matlab_class1_erp['ChannelPairNamesBankAll'][0].tolist()
        channel_names_erp = [channel_names[i] + ' ERP' for i in range(len(channel_names))]
        channel_names_hgp = [channel_names[i] + ' HGP' for i in range(len(channel_names))]

        self.results = {
            'data_erp1': data_erp_class1,
            'data_erp2': data_erp_class2,
            'data_hgp1': data_hgp_class1,
            'data_hgp2': data_hgp_class1,
            'channel_names_erp': channel_names_erp,
            'channel_names_hgp': channel_names_hgp,
            'time': time,
            'fs': 15
        }

        return self.results

    def load_data_combined_class(self):
        if self.results == {}:
            self.load_data()

        labels = np.concatenate(
            [np.zeros((self.results['data_erp1'].shape[0],)),
             np.ones((self.results['data_erp2'].shape[0],))],
            axis=0)

        data_erp = np.concatenate(
            [self.results['data_erp1'], self.results['data_erp2']],
            axis=0)

        data_hgp = np.concatenate(
            [self.results['data_hgp1'], self.results['data_hgp2']],
            axis=0)

        self.results = {
            'data_erp': data_erp,
            'data_hgp': data_hgp,
            'labels': labels,
            'channel_names_erp': self.results['channel_names_erp'],
            'channel_names_hgp': self.results['channel_names_hgp'],
            'time': self.results['time'],
            'fs': 15
        }

        return self.results

    def load_raw_data(self):
        mat_file = io.loadmat(self.paths.path_dataset_raw[0])

        data = np.array(mat_file['ft_data'][0]['trial'][0])
        label = np.squeeze(np.array(mat_file['ft_data'][0]['label'][0]))
        channel_name = mat_file['ft_data'][0]['channel_name'][0]
        channel_name = [element[0] for element in channel_name[0].tolist()]
        fs = mat_file['ft_data'][0]['fsample'][0][0, 0]

        # trial info
        trial_info = mat_file['ft_data'][0]['trialinfo'][0][:, 1:-4]
        trial_info = np.delete(trial_info, 10, axis=1)
        trial_info_hdr = mat_file['ft_data'][0]['trialinfohdr'][0]
        trial_info_hdr = [element[0] for element in trial_info_hdr[0].tolist()]
        trial_info_hdr.append('Experiment')
        trial_info = pd.DataFrame(trial_info, columns=trial_info_hdr)
        trial_info['fixation_time'] = trial_info['Image Onset NEV'] - trial_info['Fixation Onset NEV']
        trial_info['display_time'] = trial_info['TimeTrial'] - trial_info['TimeImageOnset']

        expriment_index = np.array(mat_file['ft_data'][0]['exprimentIndex'][0])
        time = np.squeeze(np.array(mat_file['ft_data'][0]['time'][0][0, 0]))

        data_ieeg = IEEGData(data, fs)
        data_ieeg.time = time
        data_ieeg.channel_name = channel_name
        data_ieeg.trial_info = trial_info
        data_ieeg.label = label

        return data_ieeg


class IEEGData:
    def __init__(self, data, fs):
        self.data = data
        self.channel_name = None
        self.trial_info = None
        self.data_info = None
        self.time = None
        self.label = None
        self.fs = fs

    def extreact_continuous_data_from_epochs(self):
        num_trial, num_channel, num_sample = self.data.shape
        if self.trial_info is None:
            raise ValueError("To make continuous data from epoched data you need trial_info but it's None")

        trial_list = []
        event_indicator_list = []
        for i in range(1, num_trial + 1):
            row = self.trial_info.loc[self.trial_info['Trial number'] == i, :]
            idx = row.index[0]
            # get the time interval from fixation to next fixateio
            time_range = [-row['fixation_time'].values[0], row['display_time'].values[0]]

            # find time idx
            t_start = np.argmin(np.abs(self.time - time_range[0]))
            t_end = np.argmin(np.abs(self.time - time_range[1]))
            t_origin = np.argmin(np.abs(self.time - 0))

            # get label from data info
            label = row['ColorLev'].values[0]

            # crate indicator signal
            event_indicator = np.zeros(self.time.shape)
            event_indicator[5000:] = 1 if label == 1 else -1

            # truncated time
            trial_list.append(self.data[idx, :, t_start:t_end])
            event_indicator_list.append(event_indicator[t_start:t_end])

        continuous_data = np.concatenate(trial_list,axis=1)
        event_indicator = np.concatenate(event_indicator_list, axis=0)


class Paths:
    def __init__(self, base_path, settings):
        self.path_dataset_raw = []
        self.task = settings.task
        self.target_class = settings.target_class
        self.patient = settings.patient
        self.base_path = base_path
        self.path_model = []
        self.path_result = []
        self.path_dataset_erp = []
        self.path_dataset_hgp = []
        self.path_stored_model = None

    def create_paths(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))  # get the working directory path
        path_saved_model = dir_path + '/saved_model/'
        if Path(path_saved_model).is_dir():
            shutil.rmtree(path_saved_model)

        Path(path_saved_model + '/model').mkdir(parents=True, exist_ok=True)
        Path(path_saved_model + '/results').mkdir(parents=True, exist_ok=True)
        self.path_model.append(os.path.join(path_saved_model + '/model/'))
        self.path_result.append(os.path.join(path_saved_model + '/results/'))

        if self.target_class == 'color':
            label_class1 = 'Black'
            label_class2 = 'White'
        elif self.target_class == 'shape':
            label_class1 = 'Shape1'
            label_class2 = 'Shape2'
        elif self.target_class == 'tone':
            label_class1 = 'Tone1'
            label_class2 = 'Tone2'
        else:
            raise ValueError('The target ' + self.target_class + ' is not defined')

        self.path_dataset_erp.append(
            self.base_path + '/Data_' + self.task + '/' + self.patient[0] + label_class1 + 'LowFreq.mat')
        self.path_dataset_erp.append(
            self.base_path + '/Data_' + self.task + '/' + self.patient[0] + label_class2 + 'LowFreq.mat')

        self.path_dataset_hgp.append(
            self.base_path + '/Data_' + self.task + '/' + self.patient[0] + label_class1 + 'HighFreq.mat')
        self.path_dataset_hgp.append(
            self.base_path + '/Data_' + self.task + '/' + self.patient[0] + label_class2 + 'HighFreq.mat')

        self.path_dataset_raw.append(self.base_path + '/Data_' + self.task + '/' + self.patient[0] +
                                     'data_single_file.mat')


class Settings:
    def __init__(self, target_class='color', task='flicker'):
        self.__patients = ['p05']

        if not isinstance(target_class, str):
            raise ValueError('"target_class" must be string!')
        else:
            if target_class.lower() not in ['color', 'shape']:
                raise ValueError('"target_class" must be color or shape!')
            self.__target_class = target_class.lower()

        if task.lower() == 'flicker':
            self.task = 'Flicker'
            self.__data_labels = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P13']
        elif task.lower() == 'flickershapes':
            self.task = 'FlickerShapes'
            self.__data_labels = ['P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18']
        elif task.lower() == 'auditory':
            self.task = 'Auditory'
            self.__data_labels = ['EP01', 'EP02', 'EP03', 'EP04']
        else:
            print('No task specified with this index')

    @property
    def target_class(self):
        return self.__target_class

    @property
    def patients(self):
        return self.__patients

    @patients.setter
    def patients(self, patient_list):
        if not isinstance(patient_list, list):
            raise ValueError('"patient" should be list!')
        else:
            for p in patient_list:
                if p not in self.data_labels:
                    raise ValueError('patient ' + p + ' is not in' + self.data_labels)
            self.__patients = patient_list

    @property
    def data_labels(self):
        return self.__data_labels


def create_synthetic_data():
    # latent process
    N_sample = 200
    X_latent = np.linspace(0, 10, N_sample)
    f1 = np.sin(X_latent)
    f2 = np.cos(0.95 * X_latent)

    # observations
    N1 = 50
    N2 = 40
    X1_obs = np.random.uniform(0, 10, N1)
    X2_obs = np.random.uniform(0, 5, N2)
    Y1 = np.sin(X1_obs) + 0.2 * np.random.randn(N1)
    Y2 = np.cos(0.95 * X2_obs) + 0.2 * np.random.randn(N2)

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(X_latent, f1, 'r-', label='Channel 1')
    plt.plot(X1_obs, Y1, 'r.', ms=10)

    plt.plot(X_latent, f2, 'b-', label='Channel 2')
    plt.plot(X2_obs, Y2, 'b.', ms=10)

    plt.legend(loc=3, shadow=True, fancybox=True, prop={'size': 14})
