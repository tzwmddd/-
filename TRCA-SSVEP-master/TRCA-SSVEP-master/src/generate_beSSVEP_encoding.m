function encoding_matrix = generate_beSSVEP_encoding(base_freqs, include_void)
% Generate beSSVEP encoding matrix: all (f_L, f_R) combinations.
% With N base frequencies and void, produces (N+1)^2 targets.
%
% function encoding_matrix = generate_beSSVEP_encoding(base_freqs, include_void)
%
% Input:
%   base_freqs    : Vector of M base stimulus frequencies [Hz]
%   include_void  : Include void (0 Hz) combinations. Default = true.
%
% Output:
%   encoding_matrix : (num_targs, 2) matrix, each row = [f_L, f_R]
%
% Reference:
%   H. Liu et al., IEEE JBHI, vol. 30, no. 5, pp. 4043-4055, 2026.

if nargin < 2
    include_void = true;
end

if include_void
    values = [0, base_freqs];
else
    values = base_freqs;
end

Nv = length(values);
encoding_matrix = zeros(Nv * Nv, 2);
idx = 1;
for i = 1:Nv
    for j = 1:Nv
        encoding_matrix(idx, :) = [values(i), values(j)];
        idx = idx + 1;
    end
end
end
