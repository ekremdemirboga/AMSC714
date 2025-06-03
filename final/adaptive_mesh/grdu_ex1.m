% File: grdu_ex1.m
function g = grdu_ex1(x_coords)
% Gradient of the exact solution u(r,theta) = r^(2/3)*sin(2*theta/3) for Problem 1
% x_coords is a 2x1 vector [x_coord; y_coord]

% Extract Cartesian coordinates
x = x_coords(1);
y = x_coords(2);

% Convert Cartesian coordinates to polar coordinates
[t, r] = cart2pol(x, y); % t is angle (theta), r is radius

% Handle r=0: the gradient is singular (undefined) at r=0.
% The function u is in H^1, but its classical gradient blows up at the origin.
if r == 0
    g = [NaN; NaN]; % Represent undefined gradient as NaN (Not-a-Number)
    return;
end

% cart2pol returns theta in the interval [-pi, pi].
% Adjust theta to be in [0, 2*pi) for consistency.
if (t < 0)
  t = t + 2*pi;
end

% Partial derivative of u with respect to r
dudr = (2/3) * r^(-1/3) * sin(2*t/3);

% Gradient of r in Cartesian coordinates: grad(r) = [cos(theta); sin(theta)] = [x/r; y/r]
grad_r = [x/r; y/r]; % Note: this is x_coords/r

% Partial derivative of u with respect to theta
dudt = r^(2/3) * cos(2*t/3) * (2/3);

% Gradient of theta in Cartesian coordinates: grad(theta) = [-sin(theta)/r; cos(theta)/r] = [-y/r^2; x/r^2]
grad_theta = [-y/r^2; x/r^2];

% Chain rule: grad(u) = dudr * grad(r) + dudt * grad(theta)
g = dudr * grad_r + dudt * grad_theta;

end
