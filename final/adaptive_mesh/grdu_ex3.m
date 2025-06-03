function g = grdu_ex3(x)

[t, r] = cart2pol(x(1), x(2));
if (t<0)
  t = t + 2*pi;
end

dudr = 2/3*r^(-1/3)*sin(2*t/3);
drdx = x/r;
dudt = r^(2/3)*cos(2*t/3)*2/3;
if (x(1)==0)
  dtdx = [-1./x(2); 0];
else
  dtdx = [-x(2)/x(1); 1]/x(1)/(1+(x(2)/x(1))^2);
end

g = dudr * drdx + dudt * dtdx;