% note this function takes vectors of spike times per channel as input.
% When a channel has less spike than others, zero pad the rest of the
% entries
%
% SPIKE TIMES SHOULD BE IN MS!

function CMatrix = CTM_spiketime_wrapper(sig)

IRI = 2:1:30;  % in ms 

channel = size(sig,2);
CMatrix = zeros(channel,channel,length(IRI));

for cnt = 1:4
    
    out = spiketime2train(sig,IRI(cnt)); 
    CMatrix(:,:,cnt) = Connect_CTM(out,18,0);
    
end

% without concurrency
for cnt = 5:length(IRI)

    out = spiketime2train(sig,IRI(cnt));
    CMatrix(:,:,cnt) = Connect_CTM(out,18,1);

end

return
