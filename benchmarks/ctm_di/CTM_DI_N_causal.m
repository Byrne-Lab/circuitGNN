function [DI, H] = CTM_DI_N_causal(X,Y,Nx,D,prob)
% DI from X to Y


[col,~] = size(Y);
if (col>1)
   Y = Y'; 
end

n_data=length(Y);

% basically reassign the probabilities
pxy = prob.pxy;
py = prob.py;

n_prob = size(py,2);


% Now this is to find py_xy
py_xy = zeros(Nx,n_prob);
for i_x=1:Nx         % can be put into one block assignment >>
    py_xy(i_x,:)=sum(pxy((i_x-1)*Nx+1:i_x*Nx,:),1);
end;

% entropy of Y itself before accumulation 
temp_H = - log2(py(Y(D+1:end)+(1:Nx:numel(py)-Nx+1)));

% This is the DI without the current X term
temp_DI = temp_H +log2(py_xy(Y(D+1:end)+(1:Nx:numel(py_xy)-Nx+1)));


% DI vector for output
H = sum(temp_H)/(n_data-D);
DI = sum(temp_DI)/(n_data-D);


return