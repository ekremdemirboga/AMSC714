function n_refined = refine_element(elem_to_refine)
% function refine_element(elem_to_refine)
% this function refines the indicated element
% and its corresponding neighbor, if the marked 
% sides coincide. 
% Otherwise, asks for a refinement of the neighbor

global mesh uh fh

n_refined = 0;

neighbour = mesh.elem_neighbours(elem_to_refine, 3);

if (neighbour > 0) % not a boundary side
  if (mesh.elem_neighbours(neighbour, 3) ~= elem_to_refine)  % not compatible
    n_refined = n_refined + refine_element(neighbour);
  end
end

n_elem = mesh.n_elem;
n_vertices = mesh.n_vertices;

% now the neighbour is compatible and comes the refinement
% the neighbour information may have changed
neighbour = mesh.elem_neighbours(elem_to_refine, 3);

v_elem = mesh.elem_vertices(elem_to_refine,:);
neighs_elem = mesh.elem_neighbours(elem_to_refine,:); 
bdries_elem = mesh.elem_boundaries(elem_to_refine,:); 

n_vertices = n_vertices + 1;
mesh.vertex_coordinates(n_vertices, :) = 0.5*(mesh.vertex_coordinates(v_elem(1),:) + mesh.vertex_coordinates(v_elem(2),:));
uh(n_vertices, :) = 0.5*(uh(v_elem(1), :) + uh(v_elem(2), :));
fh(n_vertices, :) = 0.5*(fh(v_elem(1), :) + fh(v_elem(2), :));
 
if (neighbour == 0) % marked side is at the boundary
  n_elem = n_elem + 1;
  mesh.elem_vertices(elem_to_refine, :) = [v_elem(3) v_elem(1) n_vertices];
  mesh.elem_vertices(n_elem, :)         = [v_elem(2) v_elem(3) n_vertices];

  % update neighbour information
  mesh.elem_neighbours(elem_to_refine, :) = [neighs_elem(3) n_elem neighs_elem(2)];
  mesh.elem_neighbours(n_elem, :) = [elem_to_refine neighs_elem(3) neighs_elem(1)];
  % update boundary information
  mesh.elem_boundaries(elem_to_refine, :) = [bdries_elem(3) 0 bdries_elem(2)];
  mesh.elem_boundaries(n_elem, :) = [0 bdries_elem(3) bdries_elem(1)];

  % update neighbour information of the neighbours
  if (neighs_elem(1) > 0)
    f = find(mesh.elem_neighbours(neighs_elem(1),:) == elem_to_refine);
    mesh.elem_neighbours(neighs_elem(1),f) = n_elem;
  end

  mesh.mark(n_elem) = max(mesh.mark(elem_to_refine) - 1, 0);
  mesh.mark(elem_to_refine) = max(mesh.mark(elem_to_refine) - 1, 0);
  n_refined = n_refined + 1;
else
  v_neigh = mesh.elem_vertices(neighbour,:);
  neighs_neigh = mesh.elem_neighbours(neighbour,:); 
  bdries_neigh = mesh.elem_boundaries(neighbour,:); 

  n_elem = n_elem + 2;
  mesh.elem_vertices(elem_to_refine, :) = [v_elem(3)  v_elem(1) n_vertices];
  mesh.elem_vertices(n_elem-1, :)       = [v_elem(2)  v_elem(3) n_vertices];
  mesh.elem_vertices(neighbour, :)      = [v_neigh(3) v_neigh(1) n_vertices];
  mesh.elem_vertices(n_elem, :)         = [v_neigh(2) v_neigh(3) n_vertices];

  % update neighbour information
  mesh.elem_neighbours(elem_to_refine, :) = [n_elem n_elem-1 neighs_elem(2)];
  mesh.elem_neighbours(n_elem-1, :) = [elem_to_refine neighbour neighs_elem(1)];
  mesh.elem_neighbours(neighbour, :) = [n_elem-1 n_elem neighs_neigh(2)];
  mesh.elem_neighbours(n_elem, :) = [neighbour elem_to_refine neighs_neigh(1)];
  % update boundary information
  mesh.elem_boundaries(elem_to_refine, :) = [0 0 bdries_elem(2)];
  mesh.elem_boundaries(n_elem-1, :)       = [0 0 bdries_elem(1)];
  mesh.elem_boundaries(neighbour, :)      = [0 0 bdries_neigh(2)];
  mesh.elem_boundaries(n_elem, :)         = [0 0 bdries_neigh(1)];

  % update neighbour information of the neighbours
  if (neighs_elem(1) > 0)
    f = find(mesh.elem_neighbours(neighs_elem(1),:) == elem_to_refine);
    mesh.elem_neighbours(neighs_elem(1),f) = n_elem-1;
  end
  if (neighs_neigh(1) > 0)
    f = find(mesh.elem_neighbours(neighs_neigh(1),:) == neighbour);
    mesh.elem_neighbours(neighs_neigh(1),f) = n_elem;
  end

  mesh.mark(n_elem-1) = max(mesh.mark(elem_to_refine) - 1, 0);
  mesh.mark(elem_to_refine) = max(mesh.mark(elem_to_refine) - 1, 0);

  mesh.mark(n_elem) = max(mesh.mark(neighbour) - 1, 0);
  mesh.mark(neighbour) = max(mesh.mark(neighbour) - 1, 0);
  n_refined = n_refined + 2;
end

mesh.n_elem = n_elem;
mesh.n_vertices = n_vertices;

