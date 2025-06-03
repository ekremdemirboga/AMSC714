function global_est = estimate(prob_data, adapt)
% function global_est = estimate(prob_data, adapt)
%   computes the residual type a posteriori error estimators
%   for the elliptic problem

global mesh uh

n_elem = mesh.n_elem;

mesh.estimator = zeros(n_elem, 1);

% we first compute the gradient of uh at all the elements
grduh = zeros(n_elem, 2);

% gradients of the basis functions in the reference element
grd_bas_fcts = [ -1 -1 ; 1 0 ; 0 1 ]' ;

for el = 1:n_elem
  v_elem = mesh.elem_vertices(el, :);
  v1 = mesh.vertex_coordinates( v_elem(1), : )' ;
  v2 = mesh.vertex_coordinates( v_elem(2), : )' ;
  v3 = mesh.vertex_coordinates( v_elem(3), : )' ;
  B = [ v2-v1 , v3-v1 ];

  grduh(el, :) = ( (B') \ (grd_bas_fcts*uh(v_elem)) )';
end

for el = 1:n_elem
  % interior residual
  % -div(a grad u) + b . grad u + c u - f
  % = b . grad u + c u - f
  v_elem = mesh.elem_vertices(el, :);
  v1 = mesh.vertex_coordinates( v_elem(1), : )' ;
  v2 = mesh.vertex_coordinates( v_elem(2), : )' ;
  v3 = mesh.vertex_coordinates( v_elem(3), : )' ;
  B = [ v2-v1 , v3-v1 ];
  el_area = abs(det(B))/2;
  h_T = sqrt(el_area);

  % local degrees of freedom
  uh1 = uh( v_elem(1) );
  uh2 = uh( v_elem(2) );
  uh3 = uh( v_elem(3) );

  % finite element function at the midpoints of the sides
  uh12 = (uh1 + uh2)/ 2; 
  uh23 = (uh2 + uh3)/ 2;
  uh31 = (uh3 + uh1)/ 2;

  % midpoints of the sides
  m12 = (v1 + v2) / 2;
  m23 = (v2 + v3) / 2;
  m31 = (v3 + v1) / 2;

  % values of rhs f at the midpoints of the sides
  f12 = feval(prob_data.f, m12);
  f23 = feval(prob_data.f, m23);
  f31 = feval(prob_data.f, m31);

  r12 = prob_data.b*grduh(el,:)' + prob_data.c*uh12 - f12;
  r23 = prob_data.b*grduh(el,:)' + prob_data.c*uh23 - f23;
  r31 = prob_data.b*grduh(el,:)' + prob_data.c*uh31 - f31;

  % now the jumps
  jump_res2 = 0;
  for side = 1:3
    vec = [0 0];
    if (mesh.elem_boundaries(el, side) == 0) % interior side
      vec = (grduh(el,:) - grduh(mesh.elem_neighbours(el, side),:)) ...
	    * prob_data.a;
      % the jump of the normal component is equal to the jump
      % of the full gradient 
    elseif (mesh.elem_boundaries(el, side) < 0)
      switch (side)
	case 1
	  tangential = v3 - v2;
	case 2
	  tangential = v1 - v3;
	case 3
	  tangential = v2 - v1;
      end
      % rotate clockwise to get an outer normal
      normal = [tangential(2) ; -tangential(1)]/norm(tangential);
      vec = prob_data.a*grduh(el,:)*normal - 0 ;  % only for 0 neumann so far
                 % if we want for prob_data.gN, change the 0 of the previous
                 % line by prob_data.gN(m), with m the midpoint of the segment.
    end
    jump_res2 = jump_res2 + vec*vec';
  end  

  % est(el) = sqrt(h^2 \int_T int_res2 + h \int_{dT} jump_res2
  mesh.estimator(el) = sqrt(  adapt.C(1) * h_T^2                    ...
                                  * el_area * (r12^2+r23^2+r31^2)/3 ...
			    + adapt.C(2) * h_T * h_T * jump_res2);
  
end

mesh.est_sum2 = mesh.estimator' * mesh.estimator;
mesh.max_est  = max(mesh.estimator);

global_est = sqrt(mesh.est_sum2);