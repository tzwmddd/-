% Tutorial: Binocular-Encoded SSVEP (beSSVEP) detection using
% bPRCA and FusionCA algorithms.
%
% Paper: H. Liu et al., "A Novel Binocular-Encoded SSVEP Framework
% for Efficient VR-Based Brain-Computer Interface", IEEE JBHI, 2026.
%
% Key algorithm details:
%   - bPRCA: Applies PRCA to ALL encoding frequencies for EACH target,
%     not just the target's own frequencies.
%   - FusionCA: Score(i) = sum_{f in F} R_f(i) + R_FTRCA(i)
%
% beSSVEP 4x4 encoding:
%   3 base frequencies (11, 12, 13 Hz) -> (3+1)^2 = 16 targets
%   Each target = (f_left, f_right) where f in {0, 11, 12, 13}

%% Clear workspace
clear all
close all
clc
help tutorial_fusionca

%% Set paths
addpath('../src');

%% ============================================================
%  DATA CONFIGURATION
%  ============================================================
%  Set use_paper_data = true when you have real binocular data.
%  Set use_paper_data = false to verify code with sample.mat.

use_paper_data = false;

if use_paper_data
    filename = '../data/your_beSSVEP_data.mat';  % <-- REPLACE
    fs = 1000;
    base_freqs = [11, 12, 13];
    len_gaze_s = 0.4;
    len_delay_s = 0.13;
    len_shift_s = 0.5;
    num_fbs = 5;
    is_ensemble = 1;
else
    filename = '../data/sample.mat';
    fs = 250;
    base_freqs = [8:1:15];
    len_gaze_s = 0.5;
    len_delay_s = 0.13;
    len_shift_s = 0.5;
    num_fbs = 5;
    is_ensemble = 1;
    fprintf('*** DEMO MODE: Monocular sample data ***\n');
    fprintf('*** Set use_paper_data=true for real beSSVEP ***\n\n');
end

%% Build encoding matrix
if use_paper_data
    values = [0, base_freqs];
    encoding_matrix = zeros(length(values)^2, 2);
    idx = 1;
    for i = 1:length(values)
        for j = 1:length(values)
            encoding_matrix(idx, :) = [values(i), values(j)];
            idx = idx + 1;
        end
    end
else
    encoding_matrix = [base_freqs(:), zeros(length(base_freqs), 1)];
end

num_targs = size(encoding_matrix, 1);
labels = 1:num_targs;
fprintf('Encoding: %d targets, %d base freqs\n', num_targs, length(base_freqs));

%% Prepare variables
len_gaze_smpl = round(len_gaze_s * fs);
len_delay_smpl = round(len_delay_s * fs);
len_sel_s = len_gaze_s + len_shift_s;

%% Load and segment data
load(filename);
if size(eeg, 1) ~= num_targs
    warning('Trimming data from %d to %d targets.', size(eeg, 1), num_targs);
    eeg = eeg(1:min(size(eeg, 1), num_targs), :, :, :);
    num_targs = size(eeg, 1);
    encoding_matrix = encoding_matrix(1:num_targs, :);
    labels = 1:num_targs;
end
num_blocks = size(eeg, 4);
segment_data = (len_delay_smpl + 1):(len_delay_smpl + len_gaze_smpl);
eeg_seg = eeg(:, :, segment_data, :);
fprintf('Data: %d x %d x %d x %d\n', size(eeg_seg, 1), ...
    size(eeg_seg, 2), size(eeg_seg, 3), size(eeg_seg, 4));

%% Leave-one-block-out CV
fprintf('\n=== FusionCA SSVEP Detection ===\n');
fprintf('Ensemble=%d, Length=%.2fs, FB=%d\n\n', ...
    is_ensemble, len_gaze_s, num_fbs);

for loocv_i = 1:num_blocks
    traindata = eeg_seg;
    traindata(:, :, :, loocv_i) = [];
    model = train_fusionca(traindata, fs, encoding_matrix, base_freqs, num_fbs);

    testdata = squeeze(eeg_seg(:, :, :, loocv_i));
    estimated = test_fusionca(testdata, model, is_ensemble);

    is_correct = (estimated == labels);
    accs(loocv_i) = mean(is_correct) * 100;
    itrs(loocv_i) = itr(num_targs, mean(is_correct), len_sel_s);
    fprintf('Block %d: Acc = %2.2f%%, ITR = %2.2f bpm\n', ...
        loocv_i, accs(loocv_i), itrs(loocv_i));
end

%% Summarize
mu_acc = mean(accs);
se_acc = std(accs) / sqrt(length(accs));
z_val = 1.96;
fprintf('\nMean accuracy = %2.2f %% (95%% CI: %2.2f-%2.2f %%)\n', ...
    mu_acc, mu_acc - z_val * se_acc, mu_acc + z_val * se_acc);
mu_itr = mean(itrs);
se_itr = std(itrs) / sqrt(length(itrs));
fprintf('Mean ITR = %2.2f bpm (95%% CI: %2.2f-%2.2f bpm)\n', ...
    mu_itr, mu_itr - z_val * se_itr, mu_itr + z_val * se_itr);
