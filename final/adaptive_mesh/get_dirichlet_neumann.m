function [dirichlet, neumann] = get_dirichlet_neumann;

global mesh

vertex_is_dirichlet = zeros(mesh.n_vertices, 1);

for el = 1:mesh.n_elem
  f = find(mesh.elem_boundaries(el,:) > 0);
  if (f)
    if (length(f)>1)
      % this means that all the vertices are Dirichlet
      vertex_is_dirichlet ( mesh.elem_vertices(el,:) ) = 1;
    else
      % this means that all the vertices whose local index
      % is different from 'f' are Dirichlet
      local_dirichlet_vertices = [1:3];
      local_dirichlet_vertices(f) = [];

      vertex_is_dirichlet(mesh.elem_vertices(el,local_dirichlet_vertices))=1;
    end
  end
end

dirichlet = find(vertex_is_dirichlet == 1);


neumann = zeros(mesh.n_vertices, 2);
n_neumann_segments = 0;

for el = 1:mesh.n_elem
  for vertex = 1:3
    if (mesh.elem_boundaries(el, vertex) < 0)
      % then the vertices oposite to the j-th vertex
      % form a neumann edge
      local_neumann_vertices = [1:3];
      local_neumann_vertices(vertex) = [];
      n_neumann_segments = n_neumann_segments + 1;
      neumann(n_neumann_segments, :) = mesh.elem_vertices(el, local_neumann_vertices);
    end
  end
end

if (n_neumann_segments==0)
  neumann = [];
else
  neumann(n_neumann_segments+1:end, :) = [];
end