% RLS_alg - Conventional recursive least squares
%
% Usage: [e, w] = RLS(d,u,M,ff,delta)
%
% Inputs:
% d  - the vector of desired signal samples of size Ns, ??
% u  - the vector of input signal samples of size Ns,
% M  - the number of taps. ??
% ff - forgetting factor (lambda)
% delta - initialization parameter for P
%
% Outputs:
% e - the output error vector of size Ns
% w - the last tap weights

function [e,w] = RLS_alg(d,u,M,ff,delta)

% initial values
w = zeros(M ,1);
P = eye(M) * delta; % P = delta as eye(M) is an identity matrix of size MxM

% input signal length
Ns = length(d);
u = [zeros(M-1, 1); u]; % what does it mean?

% error vector
e = zeros(Ns, 1); % HINT: aposteriori error

for i = 1:Ns
    
    % See http://www.cs.tut.fi/~tabus/course/AdvSP/21006Lect7.pdf page 24
    % Implement all the steps 2.1-2.5 (label 2.5 is used three times, nevermind that)
    
    uu = u(i+M-1:-1:i); % HINT: this is u(n) in the slide
  
    % 2.1, pi, HINT: output should be 1xM
    pi = uu'*P;
    % 2.2, gamma, HINT: output should be a scalar
    gamma = ff+pi*uu;
    % 2.3, k, HINT: output should be Mx1
    k=pi'/gamma;
    % 2.4, alpha, HINT: output should be a scalar, what is the difference
    % between this and e?
    alpha = d(i) - w'*uu;
    % 2.5, w, HINT: output should be Mx1
    w= w+k*alpha;
    % 2.5, Pprime, HINT: output should be matrix MxM
    Pprime=k*pi;
    % 2.5, P, HINT: output should be matrix MxM
    P=(P-Pprime)/ff;
    
end
end
