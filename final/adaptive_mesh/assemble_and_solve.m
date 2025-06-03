function assemble_and_solve
% function assemble_and_solve
%   assemble the discrete system and solve it
%   all the data is in the global variables
%   mesh  prob_data
%   the right-hand side is stored in 
%   the global vector  fh
%   and the discrete solution is stored in
%   the global vector  uh

global mesh prob_data uh fh

% In order to simplify, we create the dirichlet and
% neumann variables as we did in the fixed mesh case.
% That is:
%   dirichlet is a vector containing the Dirichlet vertices
%   neumann   is a matrix containing the Neumann   segments
[dirichlet, neumann] = get_dirichlet_neumann;

n_vertices = mesh.n_vertices;
n_elem = mesh.n_elem;

A  = sparse(n_vertices, n_vertices);
fh = zeros(n_vertices, 1);

% gradients of the basis functions in the reference element
grd_bas_fcts = [ -1 -1 ; 1 0 ; 0 1 ]' ;

% We loop through the elements of the mesh,
% and add the contributions of each element to the matrix A
% and the right-hand side fh

% At each element we use the cuadrature formula which uses 
% the function values at the midpoint of each side:
% \int_T  f  \approx  |T| ( f(m12) + f(m23) + f(m31) ) / 3.
% This formula is exact for quadratic polynomials

for el = 1 : n_elem
    v_elem = mesh.elem_vertices( el, : );
    
    v1 = mesh.vertex_coordinates( v_elem(1), :)' ; % coords. of 1st vertex of elem
    v2 = mesh.vertex_coordinates( v_elem(2), :)' ; % coords. of 2nd vertex of elem
    v3 = mesh.vertex_coordinates( v_elem(3), :)' ; % coords. of 3rd vertex of elem
    
    m12 = (v1 + v2) / 2; % midpoint of side 1-2
    m23 = (v2 + v3) / 2; % midpoint of side 2-3
    m31 = (v3 + v1) / 2; % midpoint of side 3-1

    % evaluation of f at the quadrature points
    f12 = feval(prob_data.f,m12);  
    f23 = feval(prob_data.f,m23);
    f31 = feval(prob_data.f,m31); 
    
    % derivative of the affine transformation from the reference
    % element onto the current element
    B = [ v2-v1  v3-v1 ];
    
    % element area
    el_area = abs(det(B)) * 0.5;

    % computation of the element load vector
    f_el = [ (f12+f31)*0.5 ; (f12+f23)*0.5 ; (f23+f31)*0.5 ] * (el_area/3);
    
    % contributions added to the global load vector
    fh( v_elem ) = fh( v_elem ) + f_el;

    Binv = inv(B);

    % computation of the element matrix
    el_mat = prob_data.a * grd_bas_fcts' * (Binv*Binv') * grd_bas_fcts * el_area ...
        + prob_data.c * el_area * [ 1/6 1/12 1/12 ; 1/12 1/6 1/12; 1/12 1/12 1/6];
  
    % contributions added to the global matrix
    A( v_elem, v_elem ) = A( v_elem, v_elem ) + el_mat;
    
end

% We now loop through the Neumann segments
% and add the integral of the basis functions against g_N
% at the corresponding position of the load vector  fh

% at each segment we use Simpson's rule
% int_a^b f \approx (b-a)/6 * ( 1 f(a) + 4 f((a+b)/2) + 1 f(b) )

if (isempty(neumann) == 0)
  n_neumann_segments = size(neumann, 1);
  for i = 1:n_neumann_segments
    v_seg = neumann(i, :);
    v1 = mesh.vertex_coordinates( v_seg(1) , : );   % coords. of 1st vertex of segment
    v2 = mesh.vertex_coordinates( v_seg(2) , : );   % coords. of 2nd vertex of segment
    m = (v1 + v2) / 2;
    
    segment_length = norm(v2-v1);
    
    g1 = feval(prob_data.gN,v1);
    g2 = feval(prob_data.gN,v2);
    gm = feval(prob_data.gN,m);
    f_seg = [ g1+2*gm ;  2*gm+g2 ] * segment_length / 6;
    
    fh( v_seg ) = fh( v_seg ) + f_seg;
  end
end


% We now impose the Dirichlet boundary conditions
% enforcing the corresponding rows of A to be  e_i
% and the right hand side to be g_D( x_i )
for i = 1:length(dirichlet)
  diri = dirichlet(i);
  A(diri,:) = zeros(1, n_vertices);
  A(diri,diri) = 1;
  fh(diri) = feval(prob_data.gD, mesh.vertex_coordinates(diri, :) );
end

% and finally we solve for u
uh = A \ fh;

end