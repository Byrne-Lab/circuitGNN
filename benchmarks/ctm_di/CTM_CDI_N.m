function  [DI, H]= CTM_CDI_N(X,Y,W,Nx,D,prob)
% DI from X to Y causally conditioned on W


% check if it is a row vector
[col,~] = size(Y);
if (col>1)
   Y = Y'; 
end

XWY = X+Nx*W+Nx^2*Y;
XW = X+Nx*W;
WY = W+Nx*Y;

n_data=length(Y);


% basically reassign the probabilities
pxwy = prob.pxwy;      % making it simpler!!!
pwy = prob.pwy;

n_prob = size(pwy,2);

% now we need to produce pxw_xyw, which is p(x_i,w_i|x^{i-1},y^{i-1},w^{i-1})
pxw_xwy = zeros(Nx^2,n_prob);
for ci = 1:Nx^2
    for cj = 1:Nx
        pxw_xwy(ci,:) = pxw_xwy(ci,:)+pxwy(ci+(cj-1)*Nx^2,:);
    end
end


% now find pw_yw, which is p(w_i|y^{i-1},w^{i-1})
pw_wy = zeros(Nx,n_prob);
for ci = 1:Nx
    for cj = 1:Nx
        pw_wy(ci,:) = pw_wy(ci,:)+pwy(ci+(cj-1)*Nx,:);
    end
end

    
temp_H = -log2(pwy(WY(D+1:end)+(1:Nx^2:numel(pwy)-Nx^2+1)))...
    + log2(pw_wy(W(D+1:end)+(1:Nx:numel(pw_wy)-Nx+1)));
temp_DI= temp_H -log2(pxw_xwy(XW(D+1:end)+(1:Nx^2:numel(pxw_xwy)-Nx^2+1)))...
    + log2(pxwy(XWY(D+1:end)+(1:Nx^3:numel(pxwy)-Nx^3+1)));
    

DI= sum(temp_DI)/(n_data-D);
H = sum(temp_H)/(n_data-D);

return
