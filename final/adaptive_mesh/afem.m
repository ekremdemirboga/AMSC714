%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AFEM in 2d for the elliptic problem
%  -div(a grad u) + b . grad u + c u = f   in Omega
%       u = gD  on Gamma_D
%   du/dn = gN  on Gamma_N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear previous figures and variables (optional, but good practice)
close all;
clearvars -except % keep breakpoints if any, or specify variables to keep
format compact; % More compact command window output

% --- Initialization Phase ---
% 1. Load problem-specific data and adaptive strategy parameters
%    This is the primary configuration file you will modify for each problem.
init_data;
% After this call, the global structs 'prob_data' and 'adapt' are populated,
% and variables like 'domain' and 'global_refinements' are set.

% 2. Declare global variables for mesh, solution (uh), and rhs (fh)
%    These are used by various functions to avoid passing large structures repeatedly.
global mesh uh fh

% 3. Initialize mesh and finite element function storage
fprintf('Loading initial mesh from domain: %s\n', domain);
mesh.elem_vertices      = load([domain '/elem_vertices.txt']);
mesh.elem_neighbours    = load([domain '/elem_neighbours.txt']);
mesh.elem_boundaries    = load([domain '/elem_boundaries.txt']);
mesh.vertex_coordinates = load([domain '/vertex_coordinates.txt']);
mesh.n_elem             = size(mesh.elem_vertices, 1);
mesh.n_vertices         = size(mesh.vertex_coordinates, 1);

% Initialize solution vector (uh) and right-hand side vector (fh)
% These will be resized if necessary during refinement.
uh = zeros(mesh.n_vertices, 1);
fh = zeros(mesh.n_vertices, 1);

% 4. Define exact solution and its gradient for error computation
%    !!! THIS IS THE MAIN PART OF AFEM.M YOU CHANGE FOR EACH PROBLEM !!!
%    For Problem 1, we use u_ex1.m and grdu_ex1.m
u_exact     = inline('u_ex1(x)', 'x');    % Exact solution function
grd_u_exact = inline('grdu_ex1(x)', 'x');  % Gradient of exact solution function
%    For a different problem, say Problem N, you would change these to:
%    u_exact     = inline('u_exN(x)', 'x');
%    grd_u_exact = inline('grdu_exN(x)', 'x');

% 5. Plot initial mesh
figure(1); % Figure for mesh plots
clf();     % Clear the figure
triplot(mesh.elem_vertices, ...
        mesh.vertex_coordinates(:,1), mesh.vertex_coordinates(:,2));
axis equal;
title('Initial Mesh');
xlabel('x'); ylabel('y');
drawnow;
fprintf('Initial mesh loaded. Press any key to continue...\n');
pause;

% 6. Perform initial global refinements if specified in init_data.m
if (global_refinements > 0)
    fprintf('Performing %d initial global refinement(s)...\n', global_refinements);
    mesh.mark = global_refinements * ones(mesh.n_elem, 1); % Mark all elements
    refine_mesh; % This function modifies global 'mesh', 'uh', 'fh'
    
    figure(1); % Update mesh plot
    clf();
    triplot(mesh.elem_vertices, ...
            mesh.vertex_coordinates(:,1), mesh.vertex_coordinates(:,2));
    axis equal;
    title(['Mesh after ', num2str(global_refinements), ' global refinement(s)']);
    xlabel('x'); ylabel('y');
    drawnow;
    fprintf('Global refinement(s) done. Press any key to start adaptive loop...\n');
    pause;
end

% --- Adaptive Loop (SOLVE -> ESTIMATE -> MARK -> REFINE) ---
iter_counter = 1;
all_n_elem = []; % To store number of elements per iteration
all_H1_err = []; % To store H1 error per iteration
all_L2_err = []; % To store L2 error per iteration
all_est    = []; % To store estimator value per iteration

fprintf('\nStarting adaptive loop...\n');
fprintf('Max iterations: %d, Tolerance for estimator: %.2e\n', adapt.max_iterations, adapt.tolerance);
fprintf('-------------------------------------------------------------------------------------------\n');
fprintf('| Iter | Elements | H1 Error   | L2 Error   | Estimator  | Max Estimator | Theta (GERS) |\n');
fprintf('|------|----------|------------|------------|------------|---------------|--------------|\n');

figure(2); % Figure for solution plots
clf();

while (1)
    % 1. SOLVE: Assemble the system and solve for uh
    assemble_and_solve; % Modifies global 'uh' and 'fh'

    % Plot the approximate solution uh
    figure(2);
    clf();
    trimesh(mesh.elem_vertices, ...
            mesh.vertex_coordinates(:,1), mesh.vertex_coordinates(:,2), ...
            uh);
    axis equal; view(3); % Use a 3D view for surface plot
    title(['Approximate Solution uh (Iteration ', num2str(iter_counter), ')']);
    xlabel('x'); ylabel('y'); zlabel('uh(x,y)');
    colorbar;
    drawnow;

    % 2. ESTIMATE: Compute the a posteriori error estimator
    est = estimate(prob_data, adapt); % Returns global estimator, updates mesh.estimator

    % Compute actual errors (if exact solution is known)
    current_H1_err = H1_err(mesh.elem_vertices, mesh.vertex_coordinates, uh, grd_u_exact);
    current_L2_err = L2_err(mesh.elem_vertices, mesh.vertex_coordinates, uh, u_exact);

    % Store data for plotting later
    all_n_elem(iter_counter) = mesh.n_elem;
    all_H1_err(iter_counter) = current_H1_err;
    all_L2_err(iter_counter) = current_L2_err;
    all_est(iter_counter)    = est;
    
    % Display iteration statistics
    % For GERS, it might be insightful to see the gamma value, but mark_elements doesn't return it.
    % We can display mesh.max_est which is used by MS and GERS.
    fprintf('| %4d | %8d | %.6e | %.6e | %.6e | %.6e |\n',...
           iter_counter, mesh.n_elem, current_H1_err, current_L2_err, est, mesh.max_est);

    % 3. Check stopping criteria
    if ((iter_counter >= adapt.max_iterations) || (est < adapt.tolerance))
        fprintf('-------------------------------------------------------------------------------------------\n');
        if (iter_counter >= adapt.max_iterations)
            fprintf('Stopping: Maximum number of iterations (%d) reached.\n', adapt.max_iterations);
        end
        if (est < adapt.tolerance)
            fprintf('Stopping: Estimator (%.2e) is below tolerance (%.2e).\n', est, adapt.tolerance);
        end
        break; % Exit the adaptive loop
    else
        % 4. MARK: Mark elements for refinement
        mark_elements(adapt); % Modifies mesh.mark based on adapt.strategy

        % 5. REFINE: Refine the mesh
        % refine_mesh uses mesh.mark and updates global 'mesh', 'uh', 'fh'
        fprintf('Refining mesh for iteration %d...\n', iter_counter + 1);
        n_refined_total_in_step = refine_mesh; % n_refined_total_in_step is the number of bisections performed
        fprintf('Number of bisections performed in this refinement step: %d\n', n_refined_total_in_step);


        % Plot the new mesh
        figure(1);
        clf();
        triplot(mesh.elem_vertices, ...
                mesh.vertex_coordinates(:,1), mesh.vertex_coordinates(:,2) );
        axis equal;
        title(['New Mesh (After Iteration ', num2str(iter_counter), ')']);
        xlabel('x'); ylabel('y');
        drawnow;
        
        fprintf('Refinement done. Press any key for next iteration...\n');
        pause;
    end
    iter_counter = iter_counter + 1;
end

fprintf('\nAdaptive process finished.\n');

% --- Optional: Plot convergence history ---
if iter_counter > 1
    figure(3);
    clf;
    subplot(2,2,1);
    loglog(all_n_elem, all_H1_err, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    xlabel('Number of Elements (N)');
    ylabel('H1 Error');
    title('H1 Error Convergence');
    grid on;

    subplot(2,2,2);
    loglog(all_n_elem, all_L2_err, 'r-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
    xlabel('Number of Elements (N)');
    ylabel('L2 Error');
    title('L2 Error Convergence');
    grid on;

    subplot(2,2,3);
    loglog(all_n_elem, all_est, 'g-^', 'LineWidth', 1.5, 'MarkerFaceColor', 'g');
    xlabel('Number of Elements (N)');
    ylabel('Error Estimator');
    title('Estimator Convergence');
    grid on;

    subplot(2,2,4);
    if ~isempty(all_H1_err) && all_H1_err(1) > 0 % Avoid division by zero if error is already zero
        effectivity_index = all_est ./ all_H1_err;
        semilogx(all_n_elem, effectivity_index, 'm-d', 'LineWidth', 1.5, 'MarkerFaceColor', 'm');
        xlabel('Number of Elements (N)');
        ylabel('Effectivity Index (Est / H1 Error)');
        title('Effectivity Index');
        grid on;
        ylim([0 max(2, max(effectivity_index))]); % Adjust y-axis for better visualization
    end
    sgtitle('Convergence History'); % Super title for the subplots
    drawnow;
end
