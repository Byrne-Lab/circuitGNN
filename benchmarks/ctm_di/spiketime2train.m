% this function will convert spike times to spike trains based 
% each channel is a column vector

function output = spiketime2train(input,IRI)

% extended length after the last spike
ext = 20;

outlen = ceil(max(max(input))/IRI)+ext;

[~,nchan] = size(input);

output = zeros(outlen,nchan);

% map inputs into the bin
input = ceil(input/IRI);

% now go through all the channels and fill in the ones
for cnt = 1:nchan

    % remove 0s from the spike time, if any
    ind = input(input(:,cnt)>0,cnt);
    output(ind,cnt)=1;
end

return