% (March 03, 2017)
% 
% Author:
% Jordan Frecon (jordan.frecon@ens-lyon.fr) 
% 
% Contributors:
% Nelly Pustelnik (nelly.pustelnik@ens-lyon.fr)
% Patrice Abry (patrice.abry@ens-lyon.fr)
% 
% This software is governed by the CeCILL license under French law and
% abiding by the rules of distribution of free software.  You can  use,
% modify and/ or redistribute the software under the terms of the CeCILL
% license as circulated by CEA, CNRS and INRIA at the following URL
% "http://www.cecill.info".
% 
% As a counterpart to the access to the source code and  rights to copy,
% modify and redistribute granted by the license, users are provided only
% with a limited warranty  and the software's author,  the holder of the
% economic rights,  and the successive licensors  have only  limited
% liability.
% 
% In this respect, the user's attention is drawn to the risks associated
% with loading,  using,  modifying and/or developing or reproducing the
% software by the user in light of its specific status of free software,
% that may mean  that it is complicated to manipulate,  and  that  also
% therefore means  that it is reserved for developers  and  experienced
% professionals having in-depth computer knowledge. Users are therefore
% encouraged to load and test the software's suitability as regards their
% requirements in conditions enabling the security of their systems and/or
% data to be ensured and,  more generally, to use and operate it in the
% same conditions as regards security.
% 
% The fact that you are presently reading this means that you have had
% knowledge of the CeCILL license and that you accept its terms.
%
%--------------------------------------------------------------------------
% Sparse Support Vector Machine for imbalanced class sizes 
%                                                                         
% For theoretical aspects please refer to :                               
% J. Spilka, J. Frecon, R.F. Leonarduzzi, N. Pustelnik, P. Abry, and M. Doret,
% Sparse Support Vector Machine for Intrapartum Fetal Heart Rate Classification, 
% Accepted to IEEE Journal of Biomedical and Health Informatics, 2016.                             
%--------------------------------------------------------------------------
%
% Sparse Support Vector Machine for imbalanced class sizes
%
% [w,b,crit] = sparseSVM (x,z,C) returns the vector 'w' and the real 'b' which
% minimize the sparse SVM, i.e.
%
%       min_(w,b) = ||w||_1 + C_P \sum_{n | z_n==+1} max(0, 1 - (x_n'*w + b))^2
%                           + C_N \sum_{n | z_n==-1} max(0, 1 + (x_n'*w + b))^2
%
%       where C_P = C and C_N = #{n | z_n==+1} / (N - #{n | z_n==+1})
%
% Note: the data fidelity term is divided into two terms in order that both classes (i.e., z=-1 & z=+1) contribute equally.
%
% INPUT 
%   - 'x' (subjects) K-by-N vector 
%   - 'z' (labels)   1-by-N vector 
%   - 'C' (trade-off between data fidelity & sparsity) real positive number 
%
% OUTPUT
%   - 'w' (unitary normal vector) K-by-1 vector 
%   - 'b' (offset) real number 
%   - 'crit' (objective function) #iterations-by-1 vector
%
% DEPENDENCY
%   - 'prox_L1.m'
%
% Version: 03-March-2017


function [w, b, crit] = sparseSVM (x,z,C)

%% Common data
% - Parameters
ind_P       = find(z==-1);
ind_N       = find(z==+1);
N_P         = length(ind_P);
N_N         = length(ind_N);
x_P         = x(:,ind_P);
x_N         = x(:,ind_N);
C_P         = C;
C_N         = (N_P/N_N) * C;

% - Algorithm Parameters
L           = 2*C_P*sum( sum( x_P.^2 ) ) + 2*C_N*sum( sum( x_N.^2 ) ) + 2*C_P*N_P + 2*C_N*N_N;
eps         = 10^-11;


% - Data Fidelity: Square Hinge loss
g.grad  = @(w, b)  2*C_P*x_P*max(0, 1 - b + x_P'*w ) - 2*C_N*x_N*max(0, 1 + b - x_N'*w );
h.grad  = @(w, b) - 2*C_P*sum(max(0, 1 - b + x_P'*w )) + 2*C_N*sum(max(0, 1 + b - x_N'*w ));
g.crit  = @(w, b) C_P*sum(max(0,1 - b + x_P'*w).^2)    + C_N*sum(max(0,1+b-x_N'*w).^2);


% - Regularization: l1
f.prox  = @(w,gamma) prox_L1(w, gamma);
f.crit  = @(w) sum(abs(w));


% - Initialization
K           = size(x,1);
N           = size(x,2);
w           = zeros(K,1);
b           = 3*mean(mean(x));%2;
crit        = zeros(1);
critm       = g.crit(w,b) + f.crit(w) ;



%% Forward-Backward

flag        = false;
gamma       = 1.99/L;
j           = 1;

while ~flag

    % - Algorithm
    b0      = b - gamma*h.grad(w,b);
    w       = f.prox(w-gamma*g.grad(w,b),gamma);
    
    % - Objective function
    crit(j) = g.crit(w,b0) +f.crit(w) ;
    b       = b0;
    
    % - Stopping criterion
    if abs(crit(j) - critm)/crit(j) < eps
        flag= true;
    end
    
    critm   = crit(j);
    j       = j + 1;

    critm
    pause
end



%% Normalization
b = b/norm(w);
w = w/norm(w);