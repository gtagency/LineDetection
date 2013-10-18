function [ d ] = rad2deg( r )
    % if this function exists, call it (this is to maintain parity with the
    % original code if this fn exists)
    if exist('radtodeg', 'builtin') ~= 0
        d = radtodeg(r);
    else
        % otherwise, its a simple calculation
        d = r * 180/pi;
    end

end

