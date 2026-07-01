function results = test_fusionca(eeg, model, is_ensemble)
% Test FusionCA-based SSVEP classification.
%
% For each candidate target i:
%   bPRCA score:  R_bPRCA(i) = sum_{f in F} R_f(i)
%   FusionCA:     Score(i) = R_bPRCA(i) + R_FTRCA(i)
%
% where R_f(i) is the Pearson correlation between the test data
% and the tiled PRC template at frequency f, both projected
% through the target i's frequency-f-specific spatial filter.
%
% function results = test_fusionca(eeg, model, is_ensemble)
%
% Input:
%   eeg          : (# targets, # channels, # samples) test data
%   model        : Trained FusionCA model (from train_fusionca.m)
%   is_ensemble  : 0 -> per-target filters, 1 -> ensemble (default: 1)
%
% Output:
%   results      : (1 x # targets) estimated target indices
%
% Reference:
%   H. Liu et al., IEEE JBHI, vol. 30, no. 5, pp. 4043-4055, 2026.

if ~exist('is_ensemble', 'var') || isempty(is_ensemble)
    is_ensemble = 1;
end
if ~exist('model', 'var')
    error('test_fusionca:NoModel', 'Trained model required.');
end

fb_coefs = (1:model.num_fbs).^(-1.25) + 0.25;
[num_targs, ~, num_smpls] = size(eeg);

for targ_i = 1:num_targs
    test_tmp = squeeze(eeg(targ_i, :, :));

    for fb_i = 1:model.num_fbs
        testdata = filterbank(test_tmp, model.fs, fb_i);

        for class_i = 1:model.num_targs
            % ---- bPRCA: sum of frequency-specific correlations ----
            R_bPRCA = 0;
            for freq_i = 1:model.num_freqs
                freq = model.freqs(freq_i);
                Rf = bprca_freq_corr(testdata, model, fb_i, class_i, ...
                    freq_i, freq, num_smpls, is_ensemble);
                R_bPRCA = R_bPRCA + Rf;
            end

            % ---- FTRCA: full-trial correlation ----
            traindata = squeeze(model.trains_TRCA(class_i, fb_i, :, :));
            if ~is_ensemble
                w_trca = squeeze(model.W_TRCA(fb_i, class_i, :));
            else
                w_trca = squeeze(model.W_TRCA(fb_i, :, :))';
            end
            r_tmp = corrcoef(testdata' * w_trca, traindata' * w_trca);
            R_TRCA = r_tmp(1, 2);

            % ---- FusionCA: simple sum (Equation 12 in paper) ----
            r(fb_i, class_i) = R_bPRCA + R_TRCA; %#ok<AGROW>
        end
    end

    rho = fb_coefs * r;
    [~, tau] = max(rho);
    results(targ_i) = tau;
end
end

function Rf = bprca_freq_corr(testdata, model, fb_i, class_k, ...
    freq_i, freq, n_smpls_test, is_ensemble)
% Compute frequency-specific bPRCA correlation R_f(i).

if freq <= 0
    Rf = 0;
    return;
end

tmpl = model.template_bPRCA{class_k, fb_i, freq_i};
if isempty(tmpl) || all(tmpl(:) == 0)
    Rf = 0;
    return;
end

[~, L_tmpl] = size(tmpl);

if is_ensemble
    w = squeeze(model.W_bPRCA(fb_i, :, freq_i, :));
    w = reshape(w, [], size(model.W_bPRCA, 4))';
else
    w = squeeze(model.W_bPRCA(fb_i, class_k, freq_i, :));
end

n_reps = ceil(n_smpls_test / L_tmpl);
tiled = repmat(tmpl, 1, n_reps);
tiled = tiled(:, 1:n_smpls_test);

r_tmp = corrcoef(testdata' * w, tiled' * w);
Rf = r_tmp(1, 2);
end
