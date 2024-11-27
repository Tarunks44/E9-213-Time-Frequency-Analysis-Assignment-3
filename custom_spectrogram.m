% Function to compute spectrogram using STFT
function [S, f, t] = custom_spectrogram(x, fs, window_length, overlap, nfft)
    % x: input signal
    % fs: sampling frequency
    % window_length: length of the window in samples
    % overlap: number of overlapping samples
    % nfft: number of FFT points

    % Create Hamming window
    window = hamming(window_length);

    % Calculate hop size
    hop_size = window_length - overlap;

    % Calculate number of time frames
    num_frames = floor((length(x) - overlap) / hop_size);

    % Initialize spectrogram matrix
    S = zeros(nfft/2 + 1, num_frames);

    % Compute STFT
    for i = 1:num_frames
        % Extract frame
        start_index = (i-1) * hop_size + 1;
        frame = x(start_index : start_index + window_length - 1);
        
        % Apply window
        windowed_frame = frame .* window;
        
        % Compute FFT
        X = fft(windowed_frame, nfft);
        
        % Keep only positive frequencies
        S(:, i) = abs(X(1:nfft/2 + 1));
    end

    % Compute time and frequency vectors
    t = (0:num_frames-1) * hop_size / fs;
    f = (0:nfft/2) * fs / nfft;
end
