close all;clear;clc;
%% Parameters
fs = 1000;  % Sampling frequency in Hz
N = 1000;   % Number of samples
n = 0:N-1;  % Time index
A = 1;      % Amplitude of the signal
window_length = 256;  % Length of the window for spectrogram
overlap = 128;  % Overlap for spectrogram
nfft = 512;     % Number of FFT points

% Problem 1: Chirp Signal (Frequency from 0.1 Hz to 0.25 Hz)
f0 = 0.1;  % Start frequency of chirp (Hz)
f1 = 0.25; % End frequency of chirp (Hz)
t = n / fs;  % Time vector in seconds
signal1 = chirp(t, f0, t(end), f1);  % Generate chirp signal

% Plot time waveform for Problem 1 (Chirp Signal)
figure;
plot(t, signal1);
title('Time Waveform: Problem 1 (Chirp Signal 0.1 Hz to 0.25 Hz)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Plot frequency (f_i[n]) for Problem 1
figure;
f = linspace(0.1, 0.25, N);  % Frequency increases linearly from 0.1 Hz to 0.25 Hz
plot(n, f);
title('Frequency vs Time: Problem 1 (Chirp Signal)');
xlabel('n →');
ylabel('Frequency →');
grid on;

% 2D Spectrogram for Problem 1 (Chirp Signal)
[S1, f1_spect, t1_spect] = custom_spectrogram(signal1, fs, window_length, overlap, nfft);

figure;
imagesc(t1_spect, f1_spect, 10*log10(S1));
axis xy;
title('2D Spectrogram: Problem 1 (Chirp Signal 0.1 Hz to 0.25 Hz)');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% 3D Spectrogram for Problem 1 (Chirp Signal)
figure;
surf(t1_spect, f1_spect, 10*log10(S1), 'EdgeColor', 'none');
view(60, 45);  % 45 to 65 degree rotation
title('3D Spectrogram: Problem 1 (Chirp Signal 0.1 Hz to 0.25 Hz)');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Power/Frequency (dB/Hz)');
colorbar;
grid on;

% Problem 2: Signal based on phi_i[n] and s[n]
f2 = linspace(0.1, 0.5, N);  % Linear frequency
phi2 = cumsum(f2);  % Compute phi_i[n] as cumulative sum
signal2 = A * cos(2 * pi * phi2);  % Generate signal s[n] = A * cos(2*pi*phi_i[n])

% Plot time waveform for Problem 2
figure;
plot(n, signal2);
title('Time Waveform: Problem 2 (Based on phi_i[n])');
xlabel('n (samples)');
ylabel('Amplitude');
grid on;

% 2D Spectrogram for Problem 2
[S2, f2_spect, t2_spect] = custom_spectrogram(signal2, fs, window_length, overlap, nfft);

figure;
imagesc(t2_spect, f2_spect, 10*log10(S2));
axis xy;
title('2D Spectrogram: Problem 2 (Based on phi_i[n])');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% 3D Spectrogram for Problem 2
figure;
surf(t2_spect, f2_spect, 10*log10(S2), 'EdgeColor', 'none');
view(65, 50);  % 45 to 65 degree rotation
title('3D Spectrogram: Problem 2 (Based on phi_i[n])');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
zlabel('Power/Frequency (dB/Hz)');
colorbar;
grid on;

%% Problem 2
% Main Program: Plot Waveform, 2D Spectrogram, and Separate 3D Spectrogram for multiple inputs
close all;clear;clc;
% List of input files and their respective sampling frequencies
files = {'Gravitational-Wave.hdf5', ...
         'Speech-Female.wav', ...
         'Speech-Male.mp3', ...
         'dog-bark-1.wav', ...
         'dog-bark-2.wav', ...
         'Perfect-Violin.wav', ...
         'ecg.mat', ...
         'eeg.set'};  % Use the correct '.set' EEG file

% Specify the sampling frequency for each signal. For some files (like wav and mp3), fs will be loaded automatically
fs_values = [4096, ...  % Gravitational Wave (sampling frequency is fixed at 4096 Hz)
             NaN,  ...  % Female Voice (fs will be loaded from the file)
             NaN,  ...  % Male Voice (fs will be loaded from the file)
             NaN,  ...  % Dog Bark 1 (fs will be loaded from the file)
             NaN,  ...  % Dog Bark 2 (fs will be loaded from the file)
             NaN,  ...  % Violin (fs will be loaded from the file)
             500,  ...  % ECG Signal (assumed to be 500 Hz)
             NaN];      % EEG (fs will be loaded from the .set file)

% Parameters for FFT and spectrogram
window_length = 256; % Window size for FFT
noverlap = 128;      % Overlap between successive windows
nfft = 512;          % Number of FFT points

% Loop through each file and process
for i = 1:length(files)
    % Load the signal based on file type (wav, mp3, hdf5, mat, set)
    [signal, fs] = load_signal(files{i}, fs_values(i));
    
    % If the signal is empty, display a message and continue to the next file
    if isempty(signal)
        disp(['Could not load signal for: ', files{i}]);
        continue;
    end

    % Plot Waveform and 2D Spectrogram in the same figure
    figure;
    
    % Plot the time-domain waveform of the signal
    t = (0:length(signal)-1) / fs;  % Time vector based on the sampling frequency
    subplot(2, 1, 1);
    plot(t, signal);  % Plot the waveform
    title(['Waveform of ', files{i}]);  % Title indicates the current file being processed
    xlabel('Time (s)');  % X-axis is time in seconds
    ylabel('Amplitude');  % Y-axis is amplitude of the signal

    % Compute the 2D Spectrogram using the custom spectrogram function
    % This computes the Short-Time Fourier Transform (STFT) of the signal
    [S, f, t_s] = custom_spectrogram(signal, fs, window_length, noverlap, nfft);
    
    % Plot the 2D Spectrogram
    subplot(2, 1, 2);
    imagesc(t_s, f, 10*log10(S));  % Convert the magnitude to dB using 10*log10(S)
    axis xy;  % Ensure the axes are displayed correctly (time on X-axis, frequency on Y-axis)
    title(['2D Spectrogram of ', files{i}]);  % Title indicates the current file's spectrogram
    xlabel('Time (s)');  % X-axis is time in seconds
    ylabel('Frequency (Hz)');  % Y-axis is frequency in Hz
    colorbar;  % Display a colorbar showing the magnitude in dB

    % Create a separate figure for the 3D spectrogram
    figure;
    
    % Plot the 3D Spectrogram
    surf(t_s, f, 10*log10(S), 'EdgeColor', 'none');  % The Z-axis represents magnitude in dB
    axis tight;  % Adjust the axes to fit the data tightly
    
    % Set the view angle between 45 and 65 degrees to enhance the 3D effect
    view(45, 60);
    
    title(['3D Spectrogram of ', files{i}]);  % Title of the 3D spectrogram
    xlabel('Time (s)');  % X-axis is time in seconds
    ylabel('Frequency (Hz)');  % Y-axis is frequency in Hz
    zlabel('Power (dB)');  % Z-axis is the magnitude in decibels
    colorbar;  % Display a colorbar for the Z-axis (magnitude in dB)

    % Pause between each signal (optional, gives time to view the plot before moving on to the next file)
    pause(1);
end


%% Problem 3
close all;clear;clc;
% Main Program: Plot 3D Narrowband and Wideband Spectrograms for multiple inputs

% List of input files and their respective sampling frequencies
files = {'Gravitational-Wave.hdf5', ...
         'Speech-Female.wav', ...
         'Speech-Male.mp3', ...
         'dog-bark-1.wav', ...
         'dog-bark-2.wav', ...
         'Perfect-Violin.wav', ...
         'ecg.mat', ...
         'eeg.set'};  % Use the correct '.set' EEG file

% Sampling frequencies for the corresponding signals
fs_values = [4096, ...  % Gravitational Wave (sampling frequency is fixed at 4096 Hz)
             NaN,  ...  % Female Voice (fs will be loaded from the file)
             NaN,  ...  % Male Voice (fs will be loaded from the file)
             NaN,  ...  % Dog Bark 1 (fs will be loaded from the file)
             NaN,  ...  % Dog Bark 2 (fs will be loaded from the file)
             NaN,  ...  % Violin (fs will be loaded from the file)
             500,  ...  % ECG Signal (assumed sampling frequency, update if necessary)
             NaN];      % EEG sampling frequency (will be loaded from the .set file)

% Parameters for narrowband and wideband spectrograms
narrowband_window_length = 1024;  % Larger window size for better frequency resolution (narrowband)
wideband_window_length = 32;      % Smaller window size for better time resolution (wideband)
noverlap_wideband = 30;           % Increased overlap for wideband spectrogram
noverlap_narrowband = 128;        % Overlap for narrowband
nfft_narrowband = 512;            % FFT size for narrowband spectrogram
nfft_wideband = 256;              % FFT size for wideband spectrogram

% Loop through each file and process
for i = 1:length(files)
    % Load the signal based on file type
    [signal, fs] = load_signal(files{i}, fs_values(i));
    
    if isempty(signal)
        disp(['Could not load signal for: ', files{i}]);
        continue;
    end

    % Plot 3D Narrowband Spectrogram
    figure;
    [S_narrow, f_narrow, t_narrow] = custom_spectrogram(signal, fs, narrowband_window_length, noverlap_narrowband, nfft_narrowband);
    surf(t_narrow, f_narrow, 10*log10(S_narrow), 'EdgeColor', 'none');
    axis tight;
    view(45, 60);  % Set view angle for 3D plot
    title(['3D Narrowband Spectrogram of ', files{i}]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    zlabel('Power (dB)');
    colorbar;
    
    % Plot 3D Wideband Spectrogram
    figure;
    [S_wide, f_wide, t_wide] = custom_spectrogram(signal, fs, wideband_window_length, noverlap_wideband, nfft_wideband);
    surf(t_wide, f_wide, 10*log10(S_wide), 'EdgeColor', 'none');
    axis tight;
    view(45, 60);  % Set view angle for 3D plot
    title(['3D Wideband Spectrogram of ', files{i}]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    zlabel('Power (dB)');
    colorbar;
    
    % Pause between each signal (optional)
    pause(1);
end

% Custom function to load signals based on file types
function [signal, fs] = load_signal(filename, fs)
    [~, ~, ext] = fileparts(filename);
    
    switch lower(ext)
        case '.wav'
            [signal, fs] = audioread(filename);  % Read the .wav file
            signal = signal(:,1);  % Take only the first channel in case of stereo

        case '.mp3'
            [signal, fs] = audioread(filename);  % Read the .mp3 file
            signal = signal(:,1);  % Take only the first channel in case of stereo

        case '.hdf5'
            % For gravitational wave data, load the strain dataset
            signal = h5read(filename, '/strain/Strain');  % Load strain data
            fs = 4096;  % The sampling frequency is fixed at 4096 Hz for this dataset

        case '.mat'
            % Load the .mat file (could be ECG or EEG data)
            data = load(filename);  % Load the .mat file into a structure
            disp('Loaded .mat file. Available variables:');
            disp(fieldnames(data));  % Display the variable names in the .mat file
            
            if isfield(data, 'val')  % ECG data case (assuming the variable is 'val')
                signal = double(data.val(:));  % Convert the signal to a column vector
            elseif isfield(data, 'EEG_signal')  % EEG data case (assuming the variable is 'EEG_signal')
                signal = double(data.EEG_signal(:));  % Convert the signal to a column vector
                fs = data.EEG_fs;  % Get the sampling frequency from the .mat file
            else
                error('Signal not found in the .mat file.');
            end

        case '.set'
            % Load EEG data using EEGLAB's pop_loadset function
            EEG = pop_loadset('filename', filename);  % Load the .set file
            signal = EEG.data(1, :);  % Take only the first channel of EEG data
            fs = EEG.srate;  % Get the sampling frequency from the EEG data

        otherwise
            % Unsupported file type
            disp(['Unsupported file type: ', ext]);
            signal = [];
            fs = [];
    end
end

% Custom spectrogram function (based on Short-Time Fourier Transform, STFT)
function [S, f, t] = custom_spectrogram(x, fs, window_length, overlap, nfft)
    % x: input signal
    % fs: sampling frequency
    % window_length: length of the window in samples
    % overlap: number of overlapping samples
    % nfft: number of FFT points

    % Create a Hamming window of the specified length
    window = hamming(window_length);

    % Calculate the hop size (the number of samples between successive windows)
    hop_size = window_length - overlap;

    % Calculate the number of time frames for the spectrogram
    num_frames = floor((length(x) - overlap) / hop_size);

    % Initialize the spectrogram matrix (magnitude of FFT results)
    S = zeros(nfft/2 + 1, num_frames);

    % Loop over each time frame and compute the FFT
    for i = 1:num_frames
        % Extract a frame from the signal (starting at index (i-1)*hop_size)
        start_index = (i-1) * hop_size + 1;
        frame = x(start_index : start_index + window_length - 1);
        
        % Apply the Hamming window to the frame
        windowed_frame = frame .* window;
        
        % Compute the FFT of the windowed frame
        X = fft(windowed_frame, nfft);
        
        % Keep only the positive frequencies and compute the magnitude
        S(:, i) = abs(X(1:nfft/2 + 1));
    end

    % Compute the time and frequency vectors for the spectrogram
    t = (0:num_frames-1) * hop_size / fs;  % Time vector (based on hop size and sampling frequency)
    f = (0:nfft/2) * fs / nfft;  % Frequency vector (based on FFT size and sampling frequency)
end
