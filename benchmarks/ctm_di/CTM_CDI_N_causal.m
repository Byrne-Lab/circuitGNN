function  [DI, H]= CTM_CDI_N_causal(X,Y,W,Nx,D,prob)
% DI from X to Y causally conditioned on W
% the arguments are passed because of some historical issue


% check if it is a row vector
[col,~] = size(Y);
if (col>1)
   Y = Y'; 
end

n_data=length(Y);


% basically reassign the probabilities
pxwy = prob.pxwy;      % making it simpler!!!
pwy = prob.pwy;

n_prob = size(pwy,2);

% find p(y_i|y^{i-1},w^{i-1}) by simply taking the marginal
py_wy = zeros(Nx,n_prob);
for i_y=1:Nx         
    py_wy(i_y,:)=sum(pwy((i_y-1)*Nx+1:i_y*Nx,:),1);
end;

% now find the marginal of p(y_i|y^{i-1},x^{i-1},w^{i-1})
py_xwy = zeros(Nx,n_prob);
for i_y=1:Nx
    py_xwy(i_y,:)=sum(pxwy((i_y-1)*Nx^2+1:i_y*Nx^2,:),1);
end
 
temp_H = -log2(py_wy(Y(D+1:end)+(1:Nx:numel(py_wy)-Nx+1)));

temp_DI= temp_H +log2(py_xwy(Y(D+1:end)+(1:Nx:numel(py_xwy)-Nx+1)));
    

DI= sum(temp_DI)/(n_data-D);
H = sum(temp_H)/(n_data-D);

return
