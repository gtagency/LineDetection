function [ r ] = rad2deg( d )
    % if this function exists, call it (this is to maintain parity with the
    % original code if this fn exists)
    if exist('degtirad', 'builtin') ~= 0
        r = degtorad(d);
    else
        % otherwise, its a simple calculation
        r = d * pi/180;
    end

end

