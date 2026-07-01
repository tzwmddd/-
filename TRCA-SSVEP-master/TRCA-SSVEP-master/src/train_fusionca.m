function model = train_fusionca(eeg, fs, encoding_matrix, freqs, num_fbs)
% Train a FusionCA model: bPRCA (periodic) + FTRCA (aperiodic).
%
% bPRCA applies PRCA to ALL encoding frequencies for EACH target,
% not just the frequencies in that target's own encoding pair.
% This captures peripheral SSVEP responses and encoding-condition-
% specific neural patterns across all reused frequency units.
%
% FusionCA score (per the paper):
%   Score(i) = sum_{f in F} R_f(i) + R_FTRCA(i)
%
% function model = train_fusionca(eeg, fs, encoding_matrix, freqs, num_fbs)
%
% Input:
%   eeg              : (# targets, # channels, # samples, # trials)
%   fs               : Sampling rate [Hz]
%   encoding_matrix  : (# targets x 2), each row = [f_L, f_R]
%                      f=0 means void (no stimulus to that eye)
%   freqs            : (1 x N) vector of all base encoding frequencies
%                      e.g., [11, 12, 13] for 3-frequency beSSVEP
%   num_fbs          : # of filter bank sub-bands (default: 5)
%
% Output:
%   model : struct with fields:
%       .trains_TRCA     : (# targets, # fbs, # channels, # samples)
%       .W_TRCA          : (# fbs, # targets, # channels)
%       .W_bPRCA         : (# fbs, # targets, # freqs, # channels)
%       .template_bPRCA  : cell{# targets, # fbs, # freqs}
%       .freqs           : (1 x # freqs) base frequency set F
%       .encoding        : (# targets x 2) encoding matrix
%       .num_fbs         : scalar
%       .fs              : scalar
%       .num_targs       : scalar
%       .num_freqs       : scalar
%
% Reference:
%   H. Liu et al., IEEE JBHI, vol. 30, no. 5, pp. 4043-4055, 2026.
%
% See also:
%   test_fusionca.m, prca.m, ftrca.m, filterbank.m

if nargin < 2
    error('train_fusionca:NotEnoughInput', 'Not enough input arguments.');
end
if ~exist('num_fbs', 'var') || isempty(num_fbs)
    num_fbs = 5;
end

[num_targs, num_chans, num_smpls, ~] = size(eeg);
num_freqs = length(freqs);

trains_TRCA = zeros(num_targs, num_fbs, num_chans, num_smpls);
W_TRCA = zeros(num_fbs, num_targs, num_chans);
W_bPRCA = zeros(num_fbs, num_targs, num_freqs, num_chans);
template_bPRCA = cell(num_targs, num_fbs, num_freqs);

for targ_i = 1:num_targs
    eeg_tmp = squeeze(eeg(targ_i, :, :, :));

    for fb_i = 1:num_fbs
        eeg_fb = filterbank(eeg_tmp, fs, fb_i);

        % FTRCA component (TRCA as special PRCA: L_TRCA = Ns)
        % This treats the entire trial as one "period"
        w_trca = ftrca(eeg_fb);
        W_TRCA(fb_i, targ_i, :) = w_trca(:, 1);
        trains_TRCA(targ_i, fb_i, :, :) = squeeze(mean(eeg_fb, 3));

        % bPRCA: PRCA at EVERY encoding frequency
        for freq_i = 1:num_freqs
            [w_tmp, tmpl_tmp] = prca(eeg_fb, fs, freqs(freq_i));
            W_bPRCA(fb_i, targ_i, freq_i, :) = w_tmp(:, 1);
            template_bPRCA{targ_i, fb_i, freq_i} = tmpl_tmp;
        end
    end
end

model = struct(...
    'trains_TRCA', trains_TRCA, ...
    'W_TRCA', W_TRCA, ...
    'W_bPRCA', W_bPRCA, ...
    'template_bPRCA', {template_bPRCA}, ...
    'freqs', freqs, ...
    'encoding', encoding_matrix, ...
    'num_fbs', num_fbs, ...
    'fs', fs, ...
    'num_targs', num_targs, ...
    'num_freqs', num_freqs);
end
