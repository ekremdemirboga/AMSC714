% File: u_ex1.m
function u = u_ex1(x_coords)
% Exact solution u(r,theta) = r^(2/3)*sin(2*theta/3) for Problem 1
% x_coords is a 2x1 vector [x_coord; y_coord]

% Convert Cartesian coordinates to polar coordinates
% x_coords(1) is the x-coordinate, x_coords(2) is the y-coordinate
[t, r] = cart2pol(x_coords(1), x_coords(2)); % t is angle (theta), r is radius

% Handle the case r=0 separately
% The solution u is 0 at the origin (r=0).
if r == 0
    u = 0;
    return;
end

% cart2pol returns theta in the interval [-pi, pi].
% For consistency in problems involving specific angular domains (like the L-shape),
% it's often useful to adjust theta to be in [0, 2*pi).
if (t < 0)
  t = t + 2*pi;
end

% Calculate the exact solution
u = r^(2/3) * sin(2*t/3);

end
