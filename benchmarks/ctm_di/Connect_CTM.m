function [CMatrix,VMatrix] = Connect_CTM(Sig,D,causal,sumspike,sig_label)

% concurrent flag is passed when the original DI definition is used
% sig_label is a string for neurons

if (causal == 0)
    DIhandle = @CTM_DI_N;
    CDIhandle = @CTM_CDI_N;
elseif (causal == 1)
    DIhandle = @CTM_DI_N_causal;
    CDIhandle = @CTM_CDI_N_causal;
else
    error('causal can only be 1 or 0');
end


cut_point = 0.01;

[sig_length, channel] = size(Sig);

% this is just a simple guard
if (channel>sig_length)
   Sig = Sig'; 
   [sig_length, channel] = size(Sig);
end

% ************* filtering out very sparse cells *************
rou = triu(ones(channel),1);

% minimum spike number
if (~exist('sumspike','var'))
    sumspike = 1;
end

for cnt = 1:channel
    if sum(Sig(:,cnt))<sumspike
        rou(cnt,:)=0;
        rou(:,cnt)=0;
    end
end



% *********** Estimate all the nasty probability *********************

% first, some specification for CTW algorithm
Nx = 2;

% process those that are connected
[row,col] = find(rou);
ind_pair = [row col];  % this will be used much later

% first find the simplex probability estimate
index = unique([row;col]);

P_matrix = zeros(2,sig_length-D,channel);

% probability of single neuron
for cnt = 1:length(index)
    P_matrix(:,:,index(cnt)) = CTM_KT_offline(Sig(:,index(cnt))',Nx,D);  
end

% define the matrix that describe the speed of the synapse
VMatrix = zeros(channel);

% joint probability of two neurons
PP_matrix = zeros(4,sig_length-D,length(row));
weight_matrix = zeros(channel);
for cnt = 1:length(row)
    out1 = Sig(:,row(cnt));
    out2 = Sig(:,col(cnt));
    [PP_matrix(:,:,cnt),~,weights1,weights2] = CTM_KT_offline((out1+Nx*out2)',Nx^2,D,1);
    weight_matrix(row(cnt),col(cnt))=sum(weights1(1+causal:end));
    weight_matrix(col(cnt),row(cnt))=sum(weights2(1+causal:end));
end


% ********* Now plot the biograph!!! Excited!! ****************

CMatrix = zeros(channel,channel);

IDs = cell(channel,1);

% see if lable is defined
if (exist('sig_label','var'))
    for cnt = 1:channel
        IDs{cnt} = sig_label{cnt};
    end
else
    for cnt = 1:channel
        IDs{cnt} = num2str(cnt);
    end
end

for cnt = 1:length(row)

    out1 = Sig(:,row(cnt));
    out2 = Sig(:,col(cnt));
    
    prob.px = P_matrix(:,:,row(cnt));
    prob.py = P_matrix(:,:,col(cnt));
    prob.pxy = PP_matrix(:,:,cnt);   % please double check

    [DI12, H2]=DIhandle(out1',out2',Nx,D,prob);
    CMatrix(row(cnt), col(cnt)) = DI12/H2;
    
    % turn it around
    prob.px = P_matrix(:,:,col(cnt));
    prob.py = P_matrix(:,:,row(cnt));
    prob.pxy = PP_matrix([1 3 2 4],:,cnt);   % please double check

    [DI21, H1]=DIhandle(out2',out1',Nx,D,prob);
    CMatrix(col(cnt), row(cnt)) = DI21/H1;

%     % debug only!!!!!!!!!!!!
%     fprintf('Hy = %f, Hxy = %f\n',H2,DI12+H2);
end

% row represents source nodes and column represents sink nodes, BS!
CMatrix(CMatrix<cut_point) = 0;


% clean up a little bit...
clear prob P_matrix



% ***************** CCDI Calculation *****************
% Now we have all the connections identified, can we eliminate all the
% indirect connections?


% mask to remove 0 entries yet not affecting existing structure
mask = ones(size(CMatrix));

% looking down every single row
for cnti = 1:channel
    
    % ask, is there any connection originated from this neuron?
    ind_col = find(CMatrix(cnti,:)>0);
    num_out = length(ind_col);
    
    % if there're at least two, it's subjected to test, hoho
    if (num_out>1)
        
        % now look at each out going synpase
        for cntj = 1:num_out
            
            % what is A and B in a A->B group
            nA = cnti;   % neuron numbers
            nB = ind_col(cntj);
            
            % all the rest of the neurons in the same row to be tested
            ind_target = ind_col(ind_col ~= nB);
            
            % for every target neuron denoted by C
            for cntk = 1:length(ind_target)
                nC = ind_target(cntk);
                
                % find out what neurons are driving the target neuron
                ind_row = find(CMatrix(:,nC)>0);
                
                % if B turns out to be a neuron that drives C, calculate
                % CCDI
                if any(abs(ind_row-nB)<1e-10)
                    
                    % first estimate the joint probability of 3 neurons
                    out1 = Sig(:,nA);
                    out2 = Sig(:,nB);
                    out3 = Sig(:,nC);
                    PPP = CTM_KT_offline((out1+Nx*out2+Nx^2*out3)',Nx^3,D);
                    
                    % pull out the probability from the joint 2 neuron
                    % array, supposed that the D estimates are the same
                    if (nA < nC)
                        match_e = sum(abs(ind_pair-repmat([nA, nC],length(row),1)),2);
                        ind = match_e==0;
                        p13 = PP_matrix(:,:,ind);
                    else
                        match_e = sum(abs(ind_pair-repmat([nC, nA],length(row),1)),2);
                        ind = match_e==0;
                        p13 = PP_matrix([1 3 2 4],:,ind);
                    end
                    
                    if (nB < nC)
                        match_e = sum(abs(ind_pair-repmat([nB, nC],length(row),1)),2);
                        ind = match_e==0;
                        p23 = PP_matrix(:,:,ind);
                    else 
                        match_e = sum(abs(ind_pair-repmat([nC, nB],length(row),1)),2);
                        ind = match_e==0;                        
                        p23 = PP_matrix([1 3 2 4],:,ind);
                    end
                    
                    % calculate A->B|C
                    prob.pxwy = PPP([1 2 5 6 3 4 7 8],:);
                    prob.pwy = p23([1 3 2 4],:);    % how to find this shit?? *******
                    [DI123_ave, ~] = CDIhandle(out1',out2',out3',Nx,D,prob);                  
                    
                    % calculate A->C|B
                    prob.pxwy = PPP;
                    prob.pwy = p23;    % how to find this shit?? *******
                    [DI132_ave, ~] = CDIhandle(out1',out3',out2',Nx,D,prob);                 
                    
                    % calculate B->C|A
                    prob.pxwy = PPP([1 3 2 4 5 7 6 8],:);  % double check!!
                    prob.pwy = p13;    % how to find this shit?? *******
                    [DI231_ave, ~] = CDIhandle(out2',out3',out1',Nx,D,prob);                 

                    % now, reject connections. Notice that only one
                    % connection can be rejected at a time!!
                    avec = [nA nA nB];
                    bvec = [nB nC nC];
                    compare_vec = [CMatrix(nA,nB) CMatrix(nA,nC) CMatrix(nB,nC)];
                    % check if more than 2 connections are down! i.e.
                    % -1-1-1 = -3
                    if ((sign(DI123_ave)+sign(DI132_ave)+sign(DI231_ave))<-1)
                        [~,ind] = min(compare_vec);
                        mask(avec(ind),bvec(ind))=0;
                    % -1-1+1 = -1
                    elseif ((sign(DI123_ave)+sign(DI132_ave)+sign(DI231_ave))<1)                        
                        suspect = find([DI123_ave DI132_ave DI231_ave]<0);
                        [~,ind] = min(compare_vec(suspect));
                        mask(avec(suspect(ind)),bvec(suspect(ind)))=0;
                    elseif (DI123_ave<0)
                        mask(nA,nB) = 0;
                    elseif (DI132_ave<0)
                        mask(nA,nC) = 0;
                    elseif (DI231_ave<0)
                        mask(nB,nC) = 0;
                    end
                    
%                    fprintf('.\n')
                    
                end               
                               
            end
            
        end
        
    end
    
end

% remove zeros entries
CMatrix = CMatrix.*mask;

% output CMatrix actually includes the sign
CMatrix = CMatrix.*sign(weight_matrix);

% here's the biograph!
if (any(CMatrix(:)) && ~nargout) 
    bg1 = biograph(CMatrix,IDs,'ShowWeights','on');
    set(bg1,'EdgeFontSize',10);
    for cnt = 1:channel
       bg1.Nodes(cnt).Shape = 'circle';
       bg1.Nodes(cnt).LineWidth = 2;
       bg1.Nodes(cnt).FontSize = 12;
    end

    % recolor the arrows and resize the lines
    % sign_di = rou+rou';
    sign_di = weight_matrix';

    % convert into a vector for ease of indexting
    sign_di = sign_di(CMatrix'>0);   % Is this right??
    strength_di = reshape(CMatrix',1,[]);
    strength_di = strength_di(strength_di>0);

    num_edges = size(bg1.Edges,1);
    for cnt = 1:num_edges
        bg1.Edges(cnt).LineWidth = ceil(strength_di(cnt)*12)*0.5+0.5;
        if (sign_di(cnt)>0)
             bg1.Edges(cnt).LineColor = [1 0 0]; % red
        else
            bg1.Edges(cnt).LineColor = [0 0 1]; % blue
        end

    end

    % finally view the biograph
    view(bg1)

end


return