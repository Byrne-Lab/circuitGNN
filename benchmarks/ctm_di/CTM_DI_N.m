function [DI, H] = CTM_DI_N(X,Y,Nx,D,prob)
% DI from X to Y


[col,~] = size(Y);
if (col>1)
   Y = Y'; 
end

XY=X+Nx*Y;

n_data=length(Y);

% basically reassign the probabilities
pxy = prob.pxy;
py = prob.py;

n_prob = size(py,2);


% px_xy is a Nx times n_data matrix, calculating p(x_i|x^{i-1},y^{i-1})
px_xy = zeros(Nx,n_prob);
for i_x=1:Nx         % can be put into one block assignment >>
    px_xy(i_x,:)=pxy(i_x,:);
    for j=2:Nx
        px_xy(i_x,:)=px_xy(i_x,:)+pxy(i_x+(j-1)*Nx,:);               % this is just taking the marginal
    end;
end;

% entropy of Y itself before accumulation 
temp_H = - log2(py(Y(D+1:end)+(1:Nx:numel(py)-Nx+1)));

% DI value before accumulation, the last two terms take care of finding
% p(y_i|x^i,y^{i-1})

temp_DI= temp_H -log2(px_xy(X(D+1:end)+(1:Nx:numel(px_xy)-Nx+1))) + log2(pxy(XY(D+1:end)+(1:Nx^2:numel(pxy)-Nx^2+1))) ;


% DI vector for output
H = sum(temp_H)/(n_data-D);
DI = sum(temp_DI)/(n_data-D);


return