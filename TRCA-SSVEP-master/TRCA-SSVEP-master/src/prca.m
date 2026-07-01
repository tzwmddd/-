function [W, PRC_template] = prca(eeg, fs, stim_freq)
% Periodically Repeated Component Analysis (PRCA).
% Segments each trial into period-length windows at the stimulus
% frequency, then maximizes inter-period reproducibility via
% generalized eigenvalue decomposition.
%
% function [W, PRC_template] = prca(eeg, fs, stim_freq)
%
% Input:
%   eeg         : (# channels, # samples, # trials)
%   fs          : Sampling rate [Hz]
%   stim_freq   : Stimulus frequency [Hz] (0 or NaN = void, returns zeros)
%
% Output:
%   W           : Spatial filter (# channels x 1)
%   PRC_template: Single-period template (# channels x period_len)
%
% Reference:
%   H. Liu et al., "A Novel Binocular-Encoded SSVEP Framework for
%   Efficient VR-Based Brain-Computer Interface", IEEE JBHI, 2026.

if nargin < 3
    error('prca:NotEnoughInput', 'Not enough input arguments.');
end

[num_chans, num_smpls, num_trials] = size(eeg);

if stim_freq <= 0 || isnan(stim_freq)
    W = zeros(num_chans, 1);
    PRC_template = zeros(num_chans, 1);
    return;
end

L = round(fs / stim_freq);
P = floor(num_smpls / L);

if P < 2
    W = zeros(num_chans, 1);
    PRC_template = zeros(num_chans, 1);
    return;
end

eeg_trunc = eeg(:, 1:P * L, :);
eeg_seg = reshape(eeg_trunc, [num_chans, L, num_trials * P]);

W = ftrca(eeg_seg);
PRC_template = squeeze(mean(eeg_seg, 3));
end
