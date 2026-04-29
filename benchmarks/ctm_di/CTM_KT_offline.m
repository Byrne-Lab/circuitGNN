function [p,D0,weightXY,weightYX] = CTM_KT_offline(x,A,D,type)
% note that the weightXY matrices produce will always contain the
% concurrent part. Remove the first entry if you don't need it


if size(x,1) ~= 1
    x = x';
    % error('The input vector must be a row vector!');
end

if nargin < 4
    type = 0;
end

% length of signal
n = length(x);

% penalty factor 
pn = 1/A;
% pn = 1;  % 1 is no penalty 

% creat a cell array that contains all nodes (that have occured)
ind_cell = cell(D+1,1);
count_cell = cell(D+1,1);
V_cell = cell(D+1,1);
I_cell = cell(D,1);

% a vector of what's occuring after each context
next_x_stat = zeros(n-D,1);
context_stat = zeros(n-D,D);
for cnt = 1:n-D
    context_stat(cnt,:) = x(cnt:cnt+D-1);
    next_x_stat(cnt) = x(cnt+D);
end
power_vec = A.^(0:D-1)';
branch_ID = context_stat*power_vec;
clear context_stat

% for the leaves
[ind_cell{D+1},~,ic]=unique(branch_ID);
num_leaves = length(ind_cell{D+1});
% count for each symbol for each leave
count_cell{D+1}=zeros(num_leaves,A);
for cnt = 1:n-D
    count_cell{D+1}(ic(cnt),next_x_stat(cnt)+1)...
        =count_cell{D+1}(ic(cnt),next_x_stat(cnt)+1)+1;   % checked
end
% find the KT probability for each leaf, in log(P)
V_cell{D+1}=zeros(num_leaves,1);
for cnt = 1:num_leaves
    V_cell{D+1}(cnt)=gammaln(A/2)+sum(gammaln(count_cell{D+1}(cnt,:)+1/2))...
        -A/2*log(pi)-gammaln(sum(count_cell{D+1}(cnt,:))+A/2)+log(pn);
end

% now go up the tree to find all the counts and the probabilities
for level = D:-1:1
    
    % find out the number of nodes and how they're connected
    node_ID = floor(ind_cell{level+1}/A);
    [ind_cell{level},~,ic] = unique(node_ID);
    
    % now do the count sums
    num_nodes = length(ind_cell{level});
    count_cell{level}=zeros(num_nodes,A);
    V_cell{level}=zeros(num_nodes,1);
    I_cell{level}=zeros(num_nodes,1);
    for noden = 1:num_nodes
        count_cell{level}(noden,:)=sum(count_cell{level+1}(ic==noden,:),1);
        
        % the estimated probability at this level
        Ps = gammaln(A/2)+sum(gammaln(count_cell{level}(noden,:)+1/2))...
            -A/2*log(pi)-gammaln(sum(count_cell{level}(noden,:))+A/2)+log(pn);
        % sum of the maximized probability from children, for log is taken
        Vas = sum(V_cell{level+1}(ic==noden,:),1)+log(pn); 
        
        if Ps < Vas
            I_cell{level}(noden)=1;
            V_cell{level}(noden)=Vas;
        else
            V_cell{level}(noden)=Ps;
        end
    end
    
end

% Finding the deepest leaf after trimming, note that a subtree could
% survive while it being trimmed down the stream
% well, I can't think of better way to implement this

% also populate a cell indicating which the deepest level is for each
% context

leafID = ind_cell{D+1};
depth_vec = zeros(num_leaves,1);
for cnt = 1:num_leaves
    
    % don't have to work on level D+1 because it's all zeros
    node_ID = leafID(cnt);
    I_vec = zeros(D+1,1);    % one extra 0 in case the leave is included
   
    for level = D:-1:1
        
        node_ID = floor(node_ID/A);
        I_vec(level) = I_cell{level}(ind_cell{level}==node_ID);
        
    end

    % after surveying along the entire context, find out the depth
    % of that particular context. Amount to finding the first 0
    % The 0 found is one index too large
    depth_vec(cnt) = find(I_vec==0,1)-1;
    
end

% the deepest branch, an output for debugging
D0 = max(depth_vec);

% This is for debug purpose
if (D0==D)
   warning('D: boundary value taken') 
end

% Simply read off the value at the leaves!

% preallocate output
p = zeros(n-D,A);
gamm = 1/2;
for cnt = 1:n-D
       
    % starting from the leaves of the truncated model, with depth Dn
    Dn = depth_vec(ind_cell{D+1}==branch_ID(cnt));
    
    node_ID = floor(branch_ID(cnt)/A^(D-Dn));     % this is the branch ID for the lowest leaf!!!!!
    ind_node = find(ind_cell{Dn+1}==node_ID);
    
    % this is simply the KT estimate at the leaf!
    pm = (count_cell{Dn+1}(ind_node,:)+gamm)/sum(count_cell{Dn+1}(ind_node,:)+gamm);
  
    % update output
    p(cnt,:)=pm;
    
end

% for the sake of compatibility
p = p';


% ****************************************************************
% let's try the sign thing

if type
    if A ~= 4
        % misuse
        error('Synapse type detect: can only be used for binary applications');
    end
    [weightXY, weightYX] = sign_helper(ind_cell,I_cell,count_cell,D0);

end


return



% ********************************************************* 
% Below are all the sign stuff

function [weightXY, weightYX] = sign_helper(ind_cell,I_cell,count_cell,D0)

gamm = 1/2;
n_valid = sum(count_cell{1});
A = 4;

% This number of leaves is based solely on the deepest context
num_leaves = length(ind_cell{D0+1});

% generate a new depth_vec for the D0, only for the ease of identifying
% signs
leafID = ind_cell{D0+1};
depth_vec = zeros(num_leaves,1);
for cnt = 1:num_leaves    
    % don't have to work on level D+1 because it's all zeros
    node_ID = leafID(cnt);
    I_vec = zeros(D0+1,1);    % one extra 0 in case the leave is included   
    for level = D0:-1:1        
        node_ID = floor(node_ID/A);
        I_vec(level) = I_cell{level}(ind_cell{level}==node_ID);        
    end
    depth_vec(cnt) = find(I_vec==0,1)-1;    
end

% Here find out the parameters associated with each branch
% preallocate output
contextID = ind_cell{D0+1};
pc = zeros(num_leaves,A);

for cnt = 1:num_leaves
    
    % starting from the leaves of the truncated model, with depth Dn
    Dn = depth_vec(cnt);
    
    node_ID = floor(contextID(cnt)/A^(D0-Dn));     % this is the branch ID for the lowest leaf!!!!!
    ind_node = find(ind_cell{Dn+1}==node_ID);
    
    % this is simply the KT estimate at the leaf!
    pm = (count_cell{Dn+1}(ind_node,:)+gamm)/sum(count_cell{Dn+1}(ind_node,:)+gamm);

    % update output
    pc(cnt,:)=pm;
    
end

% for the sake of compatibility
pc = pc';


% don't include root because it's just one single node
contextXY = zeros(length(ind_cell{D0+1}),D0);
node_ID = ind_cell{D0+1};
% find the context matrix of XY
for level = D0:-1:1
    contextXY(:,level)=mod(node_ID,4);
    node_ID = floor(node_ID/4);   
end

% frequency for each context
freqXY = sum(count_cell{D0+1},2)/n_valid;

% creat the mask that says don't care
Ms = zeros(size(contextXY));
for cnt = 1:num_leaves
    Ms(cnt,1:depth_vec(cnt))=1;
end

% find the context matrix of X
contextX = mod(contextXY,2);

% mask the X context
% contextX_masked = Ms.*contextX;
% contextX_masked = contextX;    % ***** THEORETICAL CHECK!!! *****
% ***** seems to work better without masking

% Py=1 given all possible contexts
pY1 = (pc(3,:)+pc(4,:))';
% P(y=1,Cx,Cy)
Py1Cxy = pY1.*freqXY;

weightXY = zeros(1,D0);
for cnt = 1:D0
    one_ind = find(contextX(:,cnt)==1);
    if ~isempty(one_ind)
        Py1xi1 = sum(Py1Cxy(one_ind));
        Pxi1 = sum(freqXY(one_ind));    % taken into account all x_i
        weightXY(cnt) = Py1xi1/Pxi1;
    else
        weightXY(cnt) = sum(Py1Cxy);
    end
end

% now add in the concurrent value
weightXY = [count_cell{1}(4)/(count_cell{1}(2)+count_cell{1}(4)) weightXY];

weightXY = weightXY-sum(Py1Cxy);    

% find the context matrix of Y
contextY = floor(contextXY/2);

% mask the Y context
% contextY_masked = Ms.*contextY;

pX1 = (pc(2,:)+pc(4,:))';
% P(x=1,Cx,Cy)
Px1Cxy = pX1.*freqXY;

weightYX = zeros(1,D0);
for cnt = 1:D0
    one_ind = find(contextY(:,cnt)==1);
    if ~isempty(one_ind)
        Px1yi1 = sum(Px1Cxy(one_ind));
        Pyi1 = sum(freqXY(one_ind));
        weightYX(cnt) = Px1yi1/Pyi1;
    else
        weightYX(cnt) = sum(Px1Cxy);  % don't care
    end
end

% now add in the concurrent value
weightYX = [count_cell{1}(4)/(count_cell{1}(3)+count_cell{1}(4)) weightYX];

weightYX = weightYX-sum(Px1Cxy);


return

