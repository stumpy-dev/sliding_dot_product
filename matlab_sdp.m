% This function computes the sliding dot product between
% a query Q and a time series T using the FFT method.

% See TABLE I (old) here: https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf
% See MASS_V2 here: https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
% MASS_V2 is used in [DAMP_2_0.m](https://drive.google.com/file/d/1EPDhFXQ2goTJ5m_x1S84-KNpQFsRNVmf/view?usp=sharing)

function [z] = fft_sdp(Q, T)
    % Input:
    %   Q - query vector
    %   T - time series vector
    % Output:
    %   z - sliding dot product of Q and T

    m = length(Q);
    n = length(T);

    Q = Q(end:-1:1);  % Reverse the query
    Q(m+1:n) = 0;  % Append zeros

    X = fft(T);
    Y = fft(Q);
    Z = X.*Y;
    z = ifft(Z);
    z = z(m:n);

end