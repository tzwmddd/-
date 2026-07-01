function y = filterbank(eeg, fs, idx_fb)
% Filter bank design for decomposing EEG data into sub-band components [1].
%
% function y = filterbank(eeg, fs, idx_fb)
%
% Input:
%   eeg             : Input eeg data
%                     (# of channels, Data length [sample], # of trials)
%   fs              : Sampling rate
%   idx_fb          : Index of filters in filter bank analysis
%
% Output:
%   y               : Sub-band components decomposed by a filter bank.
%
% Reference:
%   [1] X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao,
%       "Filter bank canonical correlation analysis for implementing a
%       high-speed SSVEP-based brain-computer interface",
%       J. Neural Eng., vol.12, 046008, 2015.
%
% Masaki Nakanishi, 22-Dec-2017
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if nargin < 2
    error('stats:test_fbcca:LackOfInput', 'Not enough input arguments.');
end

if nargin < 3 || isempty(idx_fb)
    warning('stats:filterbank:MissingInput',...
        'Missing filter index. Default value (idx_fb = 1) will be used.');
    idx_fb = 1;
elseif idx_fb < 1 || 10 < idx_fb
    error('stats:filterbank:InvalidInput',...
        'The number of sub-bands must be 0 < idx_fb <= 10.');
end

[num_chans, num_smpls, num_trials] = size(eeg);

passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78];
stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72];
f_low = stopband(idx_fb);
f_high = 90;

y = zeros(size(eeg));

% Probe for Signal Processing Toolbox
has_toolbox = false;
try
    cheb1ord(0.5, 0.6, 3, 40);
    has_toolbox = true;
catch %#ok<CTCH>
end

if has_toolbox
    fs_half = fs / 2;
    Wp = [passband(idx_fb) / fs_half, f_high / fs_half];
    Ws = [f_low / fs_half, 100 / fs_half];
    [N, Wn] = cheb1ord(Wp, Ws, 3, 40);
    [B, A] = cheby1(N, 0.5, Wn);
    if num_trials == 1
        for ch_i = 1:num_chans
            y(ch_i, :) = filtfilt(B, A, eeg(ch_i, :));
        end
    else
        for trial_i = 1:num_trials
            for ch_i = 1:num_chans
                y(ch_i, :, trial_i) = filtfilt(B, A, eeg(ch_i, :, trial_i));
            end
        end
    end
else
    for trial_i = 1:num_trials
        for ch_i = 1:num_chans
            if num_trials == 1
                sig = squeeze(eeg(ch_i, :));
                y(ch_i, :) = fft_bpf(sig, fs, f_low, f_high);
            else
                sig = squeeze(eeg(ch_i, :, trial_i));
                y(ch_i, :, trial_i) = fft_bpf(sig, fs, f_low, f_high);
            end
        end
    end
end
end

function sig_filt = fft_bpf(sig, fs, f_low, f_high)
N = length(sig);
freqs = (0:N - 1) / N * fs;

H = zeros(1, N);
for k = 1:N
    fk = freqs(k);
    if fk >= f_low && fk <= f_high
        H(k) = 1;
    elseif fk >= (f_low - 2) && fk < f_low
        H(k) = 0.5 - 0.5 * cos(pi * (fk - (f_low - 2)) / 2);
    elseif fk > f_high && fk <= (f_high + 2)
        H(k) = 0.5 + 0.5 * cos(pi * (fk - f_high) / 2);
    end
end

X = fft(sig);
sig_filt = real(ifft(X .* H));
end
