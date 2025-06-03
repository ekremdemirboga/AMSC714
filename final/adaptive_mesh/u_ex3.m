function u = u_ex3(x)

[t, r] = cart2pol(x(1), x(2));
if (t<0)
  t = t + 2*pi;
end

u = r^(2/3)*sin(2*t/3);
