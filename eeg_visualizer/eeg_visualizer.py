import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.express as px
import scipy.fftpack as fftpack
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from eeg_visualizer.multitapper.multitaper_spectrogram_python import multitaper_spectrogram, plotly_heatmap

# import multitaper_spectrogram function from the multitaper_spectrogram_python.py file

FIGURE_HEIGHT = '720px'


class VizServer:
    def __init__(self, dataset, trial_info, channels_info):
        """

        """
        # Set spectrogram params

        self.dataset = dataset
        self.df = trial_info
        self.df_channels = channels_info

        self.init_flag = True
        self.min_nfft = 0  # No minimum nfft
        self.detrend_opt = 'constant'  # detrend each window by subtracting the average
        self.multiprocess = True  # use multiprocessing
        self.cpus = 7  # use 3 cores in multiprocessing
        self.weighting = 'unity'  # weight each taper at 1
        self.plot_on = False  # plot spectrogram
        self.return_fig = True  # do not return plotted spectrogram
        self.clim_scale = False  # do not auto-scale colormap
        self.verbose = False  # print extra info
        self.xyflip = False  # do not transpose spect output matrix
        self.init_val = False

        # Initialize
        self.fs = dataset.fs
        self.time_range = [-0.5, 1]
        self.frequency_range = [0, 25]  # Limit frequencies from 0 to 25 Hz
        self.time_bandwidth = 3  # Set time-half bandwidth
        self.num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
        self.window_params = [1, 0.25]  # Window size is 1s with step size of 0.25s

        self.time_info = None
        self.freqs = None
        self.erp_signal1 = None
        self.erp_signal2 = None
        self.psd_mean1 = None
        self.psd_mean2 = None
        self.psds = None
        self.spect_all = None
        self.spect_mean1 = None
        self.spect_mean2 = None

        # Build the app
        self.app = dash.Dash(__name__)
        self.init_val = True
        self.init_values()
        self.layout()
        self.callbacks()

    def init_values(self, min_time_input=-0.5, max_time_input=1, get_mean_tfr=False):

        t_range = [min_time_input, max_time_input]

        t_start = np.argmin(np.abs(self.dataset.time - t_range[0]))
        t_end = np.argmin(np.abs(self.dataset.time - t_range[-1]))

        time_vect = self.dataset.time[t_start:t_end]

        self.time_info = (t_start, t_end, time_vect)

        self.erp_signal1 = np.mean(self.dataset.data[self.dataset.label == 0, :, t_start:t_end], axis=0)
        self.erp_signal2 = np.mean(self.dataset.data[self.dataset.label == 1, :, t_start:t_end], axis=0)

        num_channel = self.dataset.data.shape[1]
        num_trials = self.dataset.data.shape[0]
        signal_length = self.dataset.data[:, :, t_start:t_end].shape[-1]
        ts = 1 / self.fs
        self.freqs = np.linspace(0.0, 1.0 / (2.0 * ts), signal_length // 2)
        self.psds = np.zeros((num_trials, num_channel, len(self.freqs)))
        for ch in range(num_channel):
            yf = fftpack.fft(self.dataset.data[:, ch, t_start:t_end])
            self.psds[:, ch, :] = 2.0 / signal_length * np.abs(yf[:, :signal_length // 2])

        self.psd_mean1 = np.mean(self.psds[self.dataset.label == 0, :, :], axis=0)
        self.psd_mean2 = np.mean(self.psds[self.dataset.label == 1, :, :], axis=0)

        t_start, t_end, times = self.time_info
        if get_mean_tfr == ['1']:
            self.spect_all = None
            for tr_idx in range(num_trials):
                for ch_idx in range(num_channel):
                    signal = self.dataset.data[tr_idx, ch_idx, t_start:t_end]
                    spect, stimes, sfreqs = multitaper_spectrogram(signal,self.fs,self.frequency_range,
                                                                   self.time_bandwidth,self.num_tapers,
                                                                   self.window_params, self.min_nfft,self.detrend_opt,
                                                                   self.multiprocess,self.cpus,self.weighting,
                                                                   self.plot_on,self.return_fig,self.clim_scale,
                                                                   self.verbose,self.xyflip,
                                                                   plotly_on=False,
                                                                   times=times)
                    if self.spect_all is None:
                        self.spect_all = np.zeros(shape=(num_trials, num_channel, spect.shape[0], spect.shape[1]))
                    self.spect_all[tr_idx, ch_idx, :, :] = spect/np.max(np.abs(spect))

            self.spect_mean1 = np.mean(self.spect_all[self.dataset.label == 0, :, :, :], axis=0)
            self.spect_mean2 = np.mean(self.spect_all[self.dataset.label == 1, :, :, :], axis=0)
            self.stimes = stimes
            self.sfreqs = sfreqs

    def layout(self):
        # Define app and layout
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div(
            style={
                'backgroundColor': '#1a1a1a',
                'color': '#FFFFFF',
                'fontFamily': 'Arial',
                'height': 'max-content',
                'width': 'max-content',
                'display': 'flex',
                'min-width': '100%',
                'min-height': '100%'
            },
            children=[
                # Trial table
                html.Div(
                    style={
                        'flex': '1 1 10%',
                        'padding': '20px'
                    },
                    children=[
                        html.H1(
                            children='Trials',
                            style={
                                'color': '#FFFFFF',
                                'fontSize': '36px'
                            }
                        ),
                        dash_table.DataTable(
                            id='table_trial',
                            columns=[{"name": i, "id": i} for i in self.df.columns],
                            data=self.df.to_dict('records'),
                            row_selectable='single',
                            style_table={
                                'height': '900px',
                                'width': '300px',
                                'overflowY': 'scroll',
                                'overflowX': 'scroll'
                            },
                            style_cell={
                                'backgroundColor': '#2b2b2b',
                                'color': '#FFFFFF',
                                'textAlign': 'left',
                                'fontFamily': 'Arial',
                                'padding': '5px',
                                'border': '1px solid #1a1a1a'
                            },
                            style_header={
                                'backgroundColor': '#1a1a1a',
                                'color': '#FFFFFF',
                                'textAlign': 'left',
                                'fontFamily': 'Arial',
                                'fontWeight': 'bold',
                                'padding': '5px',
                                'border': '1px solid #1a1a1a'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#3d3d3d',
                                    'border': '1px solid #1a1a1a'
                                },
                                {
                                    'if': {'state': 'selected'},
                                    'backgroundColor': '#4d4d4d',
                                    'border': '1px solid #FFFFFF'
                                }
                            ]
                        )
                    ]
                ),
                # Channel Table
                html.Div(
                    style={
                        'flex': '1 1 10%',
                        'padding': '20px'
                    },
                    children=[
                        html.H1(
                            children='Channels',
                            style={
                                'color': '#FFFFFF',
                                'fontSize': '36px'
                            }
                        ),
                        dash_table.DataTable(
                            id='table_channels',
                            columns=[{"name": i, "id": i} for i in self.df_channels.columns],
                            data=self.df_channels.to_dict('records'),
                            row_selectable='single',
                            style_table={
                                'height': '900px',
                                'width': '300px',
                                'overflowY': 'scroll',
                                'overflowX': 'scroll'
                            },
                            style_cell={
                                'backgroundColor': '#2b2b2b',
                                'color': '#FFFFFF',
                                'textAlign': 'left',
                                'fontFamily': 'Arial',
                                'padding': '5px',
                                'border': '1px solid #1a1a1a'
                            },
                            style_header={
                                'backgroundColor': '#1a1a1a',
                                'color': '#FFFFFF',
                                'textAlign': 'left',
                                'fontFamily': 'Arial',
                                'fontWeight': 'bold',
                                'padding': '5px',
                                'border': '1px solid #1a1a1a'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#3d3d3d',
                                    'border': '1px solid #1a1a1a'
                                },
                                {
                                    'if': {'state': 'selected'},
                                    'backgroundColor': '#4d4d4d',
                                    'border': '1px solid #FFFFFF'
                                }
                            ]
                        )
                    ]
                ),
                # Tabs
                html.Div(
                    style={
                        'flex': '1 1 80%',
                        'padding': '20px'
                    },
                    children=[
                        dcc.Tabs(
                            id='tabs',
                            style={
                                'color': '#000033',
                                'fontWeight': 'bold',
                                'background-color': '#111144'
                            },
                            children=[
                                dcc.Tab(
                                    label='Signals',
                                    children=[
                                        dcc.Loading(
                                            id="loading-signal",
                                            type="cube",
                                            children=[
                                                dcc.Graph(
                                                    id='signals-plot',
                                                    figure={},
                                                    style={'height': FIGURE_HEIGHT, 'width': '100%', 'flex': '1 1'}
                                                )
                                            ]
                                        )
                                    ],
                                    style={'background-color': '#000011',
                                           'border': '1px solid #222222',
                                           'color': '#FFFFFF'}
                                ),
                                dcc.Tab(
                                    label='FFT',
                                    children=[
                                        dcc.Loading(
                                            id="loading-fft",
                                            type="cube",
                                            children=[
                                                dcc.Graph(
                                                    id='histogram-plot',
                                                    figure={},
                                                    style={'height': FIGURE_HEIGHT, 'width': '100%', 'flex': '1 1'}
                                                )
                                            ]
                                        )
                                    ],
                                    style={'background-color': '#000011',
                                           'border': '1px solid #222222',
                                           'color': '#FFFFFF'}
                                ),
                                dcc.Tab(
                                    label='ُSpectrogram',
                                    children=[
                                        dcc.Loading(
                                            id="loading-spectrogram",
                                            type="cube",
                                            children=[
                                                dcc.Graph(
                                                    id='spectrogram-plot',
                                                    figure={},
                                                    style={'height': FIGURE_HEIGHT, 'width': '100%', 'flex': '1 1'}
                                                )
                                            ]
                                        )
                                    ],
                                    style={'background-color': '#000011',
                                           'border': '1px solid #222222',
                                           'color': '#FFFFFF'}
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                'padding': '20px'
                            },
                            children=[
                                html.H1(
                                    children='Settings',
                                    style={
                                        'color': '#FFFFFF',
                                        'fontSize': '36px'
                                    }
                                ),
                                # Check Button
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'flex-direction': 'row',
                                        'justify-content': 'space-between',
                                        'align-items': 'flex-start',  # added property
                                        'margin-bottom': '10px'
                                    },
                                    children=[
                                        dcc.Checklist(
                                            id='checkbutton',
                                            options=[
                                                {'label': 'Calculate Synchronous TFR (Takes time)', 'value': '1'},
                                            ],
                                            value=[],
                                            style={
                                                'font-size': '24px',  # increase font size of label
                                            },
                                            labelStyle={
                                                'display': 'block',
                                                'font-size': '20px',  # increase font size of label text
                                                'padding': '10px 10px'  # add padding to top and bottom of label
                                            },
                                            inputStyle={
                                                'width': '20px',  # increase size of checkbox
                                                'height': '20px'  # increase size of checkbox
                                            }
                                        )
                                    ]),
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'flex-direction': 'row',
                                        'justify-content': 'space-between',
                                        'align-items': 'flex-start',  # added property
                                        'margin-bottom': '10px'
                                    },
                                    children=[
                                        html.Label(
                                            children='Minimum Time  :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='min-time-input',
                                            type='number',
                                            value=-0.5,
                                            min=self.dataset.time[0],
                                            max=self.dataset.time[-1],
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        ),
                                        html.Label(
                                            children='Maximum Time  :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='max-time-input',
                                            type='number',
                                            value=1.5,
                                            min=self.dataset.time[0],
                                            max=self.dataset.time[-1],
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        ),
                                        html.Label(
                                            children='Min Frequency :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='min-freq-input',
                                            type='number',
                                            value=0,
                                            min=0,
                                            max=120,
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        ),
                                        html.Label(
                                            children='Max Frequency :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='max-freq-input',
                                            type='number',
                                            value=25,
                                            min=0,
                                            max=120,
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        )
                                    ]),
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'flex-direction': 'row',
                                        'justify-content': 'space-between',
                                        'align-items': 'flex-start',  # added property
                                        'margin-bottom': '10px'
                                    },
                                    children=[
                                        html.Label(
                                            children='Time Bandwidth:',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'}),
                                        dcc.Input(
                                            id='time-bandwidth-input',
                                            type='number',
                                            value=3,
                                            min=0,
                                            max=10,
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        ),
                                        html.Label(
                                            children='Tapers        :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='tapers-input',
                                            type='number',
                                            value=5,
                                            min=0,
                                            max=10,
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        ),
                                        html.Label(
                                            children='Window Size   :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='win-size-input',
                                            type='number',
                                            value=1,
                                            min=0,
                                            max=5,
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        ),
                                        html.Label(
                                            children='Step Size     :',
                                            style={
                                                'color': '#FFFFFF',
                                                'fontSize': '18px',
                                                'margin-left': '10px',
                                                'margin-right': '10px'
                                            }
                                        ),
                                        dcc.Input(
                                            id='step-size-input',
                                            type='number',
                                            value=0.25,
                                            min=0,
                                            max=10,
                                            style={
                                                'width': '100px',
                                                'height': '30px',
                                                'padding': '5px',
                                                'border': '1px solid #1a1a1a'
                                            }
                                        )
                                    ]
                                ),
                                html.Button(
                                    'Generate',
                                    id='generate-button',
                                    style={
                                        'width': '100%',
                                        'height': '50px',
                                        'backgroundColor': '#4CAF50',
                                        'color': '#FFFFFF',
                                        'fontSize': '20px',
                                        'border': 'none',
                                        'border-radius': '4px',
                                        'cursor': 'pointer'
                                    }
                                )]
                        )
                    ]
                )
            ]
        )

    def callbacks(self):
        # Define callbacks
        @self.app.callback(
            [
                dash.dependencies.Output('signals-plot', 'figure'),
                dash.dependencies.Output('histogram-plot', 'figure'),
                dash.dependencies.Output('spectrogram-plot', 'figure'),

            ],
            [
                dash.dependencies.Input('table_trial', 'selected_rows'),
                dash.dependencies.Input('table_channels', 'selected_rows'),
                dash.dependencies.Input('generate-button', 'n_clicks')

            ],
            [
                dash.dependencies.State('min-time-input', 'value'),
                dash.dependencies.State('max-time-input', 'value'),
                dash.dependencies.State('min-freq-input', 'value'),
                dash.dependencies.State('max-freq-input', 'value'),
                dash.dependencies.State('time-bandwidth-input', 'value'),
                dash.dependencies.State('tapers-input', 'value'),
                dash.dependencies.State('win-size-input', 'value'),
                dash.dependencies.State('step-size-input', 'value'),
                dash.dependencies.State('checkbutton', 'value')
            ]
        )
        def update_figures(selected_rows_trial,
                           selected_rows_channel,
                           n_clicks,
                           min_time_input, max_time_input,
                           min_freq_input,
                           max_freq_input,
                           time_bandwidth_input,
                           tapers_input, win_size_input,
                           step_size_input,
                           check_button_value):
            if n_clicks is not None or self.init_val is False:
                # Replace this with your actual sampling frequency
                self.init_val = True
                self.time_range = [min_time_input, max_time_input]
                self.frequency_range = [min_freq_input, max_freq_input]  # Limit frequencies from 0 to 25 Hz
                self.time_bandwidth = time_bandwidth_input  # Set time-half bandwidth
                self.num_tapers = tapers_input  # Set number of tapers (optimal is time_bandwidth*2 - 1)
                self.window_params = [win_size_input, step_size_input]  # Window size is 1s with step size of 0.25s

                if check_button_value != ['1']:
                    self.spect_all = None

                self.init_values(min_time_input=min_time_input,
                                 max_time_input=max_time_input,
                                 get_mean_tfr=check_button_value)
            t_start, t_end, times = self.time_info
            if selected_rows_channel is None or len(selected_rows_channel) == 0:
                selected_rows_channel = [0]
            if selected_rows_trial is None or len(selected_rows_trial) == 0:
                selected_rows_trial = [0]
            # data info
            selected_row = self.df.iloc[selected_rows_trial[0], :]
            selected_channel_row = self.df_channels.iloc[selected_rows_channel[0], 0]
            selected_column = selected_row.index[0]

            # Signal in time domain
            signal = self.dataset.data[selected_rows_trial[0], selected_rows_channel[0], t_start:t_end]
            mean_signal1 = self.erp_signal1[selected_rows_channel[0], :]
            mean_signal2 = self.erp_signal2[selected_rows_channel[0], :]
            signals_df = pd.DataFrame({
                'Signal': signal,
                'ERP1': mean_signal1,
                'ERP2': mean_signal2,
                'time': times
            })

            # Signal in frequency domain
            yf_abs = self.psds[selected_rows_trial[0], selected_rows_channel[0], :]
            yf_abs_mean1 = self.psd_mean1[selected_rows_channel[0], :]
            yf_abs_mean2 = self.psd_mean2[selected_rows_channel[0], :]
            fft_df = pd.DataFrame({
                'fft': yf_abs,
                'mean fft1': yf_abs_mean1,
                'mean fft2': yf_abs_mean2,
                'freq': self.freqs
            })

            # Plot the three signals with one signal in subplot 1 and two signals in subplot 2
            fig1 = make_subplots(rows=2, cols=1)
            fig1.add_trace(px.line(signals_df, x='time', y='Signal').data[0], row=1, col=1)
            fig1.add_trace(px.line(signals_df, x='time', y=['ERP1', 'ERP2'],
                                   line_group='variable',
                                   color_discrete_map={'Signal': 'green', 'ERP1': 'red', 'ERP2': 'blue'}).data[0],
                           row=2, col=1)
            fig1.add_trace(px.line(signals_df, x='time', y=['ERP1', 'ERP2'],
                                   line_group='variable',
                                   color_discrete_map={'Signal': 'green', 'ERP1': 'red', 'ERP2': 'blue'}).data[1],
                           row=2, col=1)
            # Set the y-axis range to be the same for subplot 2

            fig1.update_layout(
                template='plotly_dark',
                title=selected_channel_row,
                yaxis={'title': 'Signal'},
                yaxis2={'title': 'ERP'},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True,
                xaxis2=dict(matches='x', title='Time (S)'),
            )

            # Update histogram
            fig2 = make_subplots(rows=2, cols=1)
            fig2.add_trace(px.line(fft_df, x='freq', y='fft', color_discrete_map={'fft': 'green'}).data[0], row=1,
                           col=1)
            fig2.add_trace(px.line(fft_df, x='freq', y=['mean fft1', 'mean fft2'],
                                   line_group='variable',
                                   color_discrete_map={'fft': 'green', 'mean fft1': 'red', 'mean fft2': 'blue'}).data[
                               0], row=2,
                           col=1)
            fig2.add_trace(px.line(fft_df, x='freq', y=['mean fft1', 'mean fft2'],
                                   line_group='variable',
                                   color_discrete_map={'fft': 'green', 'mean fft1': 'red', 'mean fft2': 'blue'}).data[
                               1], row=2,
                           col=1)
            fig2.update_xaxes(range=[0, 120])
            fig2.update_layout(
                template='plotly_dark',
                title=selected_channel_row,
                yaxis={'title': 'FFT'},
                yaxis2={'title': 'Mean FFT'},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True,
                xaxis2=dict(matches='x', title='frequency (Hz)')
            )
            # Compute the multitaper spectrogram

            if check_button_value == ['1'] and self.spect_all is not None:
                tfr = self.spect_all[selected_rows_trial[0], selected_rows_channel[0], :,:]
                mean_tfr1 = self.spect_mean1[selected_rows_channel[0], :, :]
                mean_tfr2 = self.spect_mean2[selected_rows_channel[0], :, :]

                # Update histogram
                heatmap = plotly_heatmap(tfr, self.stimes, self.sfreqs)
                heatmap_mean1 = plotly_heatmap(mean_tfr1, self.stimes, self.sfreqs)
                heatmap_mean2 = plotly_heatmap(mean_tfr2, self.stimes, self.sfreqs)

                fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True)
                fig3.add_trace(heatmap, row=1, col=1)
                fig3.add_trace(heatmap_mean1, row=2, col=1)
                fig3.add_trace(heatmap_mean2, row=3, col=1)

                # set the coloraxis property of each trace to the same value
                fig3.data[0].update(coloraxis="coloraxis")
                fig3.data[1].update(coloraxis="coloraxis")
                fig3.data[2].update(coloraxis="coloraxis")

                fig3.update_layout(template='plotly_dark',
                                   xaxis1={'title': 'Time (s)'},
                                   yaxis={'title': 'Frequency (Hz)'},
                                   title=selected_channel_row,
                                   yaxis2={'title': 'Frequency (Hz)'},
                                   yaxis3={'title': 'Frequency (Hz)'},
                                   xaxis2={'title': 'Time (s)'},
                                   xaxis3={'title': 'Time (s)'})
            else:
                spect, stimes, sfreqs, fig3 = multitaper_spectrogram(signal,
                                                                     self.fs,
                                                                     self.frequency_range,
                                                                     self.time_bandwidth,
                                                                     self.num_tapers,
                                                                     self.window_params,
                                                                     self.min_nfft,
                                                                     self.detrend_opt,
                                                                     self.multiprocess,
                                                                     self.cpus,
                                                                     self.weighting,
                                                                     self.plot_on,
                                                                     self.return_fig,
                                                                     self.clim_scale,
                                                                     self.verbose,
                                                                     self.xyflip,
                                                                     plotly_on=True, times=times)
                fig3.update_layout(template='plotly_dark',
                                   xaxis={'title': 'Time (s)'},
                                   yaxis={'title': 'Frequency (Hz)'})

            return fig1, fig2, fig3

    def run_server(self, debug=False):
        self.app.run_server(debug=debug)
