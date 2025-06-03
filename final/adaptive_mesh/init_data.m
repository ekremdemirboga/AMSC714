% File: init_data.m
% Updated for Problem 1: Corner Singularity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  problem  data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Folder or directory where the domain mesh is described
% For Problem 1, this should be the L-shaped domain.
domain = 'L_shape_dirichlet'; % This should already be set if you are using the L-shape files

% Initial global refinements (can be adjusted based on experiments)
global_refinements = 1; % Default value from the provided file

global prob_data

% PDE: -div(a grad u) + b . grad u + c u = f

% Diffusion coefficient (a) of the equation
% For Poisson equation -Delta u = f, which is -div(1 * grad u) = f, so a=1.
prob_data.a = 1;

% Convection coefficient (b) of the equation (row vector)
% For Poisson equation, there is no convection term.
prob_data.b = [0.0  0.0];

% Reaction coefficient (c) of the equation
% For Poisson equation, there is no reaction term.
prob_data.c = 0.0;

% Right-hand side function f
% For Problem 1, we found f = 0.
prob_data.f = inline('0', 'x');

% Dirichlet data, function g_D
% This is the exact solution evaluated on the boundary.
% We will use the u_ex1.m function created in Step 2.
prob_data.gD = inline('u_ex1(x)','x'); % Changed from u_ex3

% Neumann data, function g_N
% Problem 1 specifies Dirichlet boundary conditions on the entire boundary.
% So, gN is not actively used if the boundary is correctly marked as Dirichlet.
prob_data.gN = inline('0', 'x'); % Placeholder, as it's a full Dirichlet problem

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  data for a posteriori estimators and adaptive strategy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global adapt


adapt.C(1) = 0.2; 
adapt.C(2) = 0.2; 

% Stop when energy error is smaller than tol = 3e-2
adapt.tolerance = 3e-2; 

% Stop either when iterations = max_iter = 34 or tolerance is met.
adapt.max_iterations = 34; 

% Marking_strategy, possible options are
% GR: global (uniform) refinement,
% MS: maximum strategy,
% GERS: guaranteed error reduction strategy (Dörfler's)
adapt.strategy = 'GERS'; % Default is GERS.

adapt.n_refine = 2; % Default from provided file

% Parameters of the different marking strategies
% Perform these experiments with threshold theta=0.5 for element marking.
adapt.MS_gamma = 0.5; 

% GERS: guaranteed error reduction strategy (Dörfler's)
% If using GERS, theta=0.5 should apply here too.
adapt.GERS_theta_star = 0.2; % Changed from 0.8 to match problem spec theta=0.5
adapt.GERS_nu = 0.1; % Default from provided file

