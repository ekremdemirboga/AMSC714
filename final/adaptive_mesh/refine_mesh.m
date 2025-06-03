function n_refined = refine_mesh
% function n_refined = refine_mesh
%   This function refines the mesh defined in the global variable
%   mesh, with the marked elements mesh.mark
%   and interpolates uh and fh

global mesh uh fh

if (max(mesh.mark)==0)
  % no elements marked, doing nothing
  return
end

% we first extend the matrices and vectors to save 
% time in memory allocation
% estimation of final number of elements
nelem_new = mesh.n_elem*2*max(mesh.mark);
mesh.elem_vertices(nelem_new,1) = 0;
mesh.elem_neighbours(nelem_new,1) = 0;
mesh.elem_boundaries(nelem_new,1) = 0;
mesh.mark(nelem_new) = 0;

% estimation of final number of vertices
% improve this computation
nvertices_new = 4*mesh.n_vertices;
mesh.vertex_coordinates(nvertices_new,1) = 0;
uh(nvertices_new,1) = 0;
fh(nvertices_new,1) = 0;

n_refined = 0;

while (max(mesh.mark) > 0)
  first_marked = min(find(mesh.mark > 0));
  n_refined = n_refined + refine_element(first_marked);
end


% we now clean the unused space
mesh.elem_vertices(mesh.n_elem+1:nelem_new,:) = [];
mesh.elem_neighbours(mesh.n_elem+1:nelem_new,:) = [];
mesh.elem_boundaries(mesh.n_elem+1:nelem_new,:) = [];
mesh.mark(mesh.n_elem+1:nelem_new) = [];

mesh.vertex_coordinates(mesh.n_vertices+1:nvertices_new,:) = [];
uh(mesh.n_vertices+1:nvertices_new,:) = [];
fh(mesh.n_vertices+1:nvertices_new,:) = [];
