To view figures only without saving, change SAVEFIG at the beginning of each file to False.

===============================================

"part_a.py"

    Plot of steady state solution for various values of right/left boundary conditions and beta values.

    Plot for truncation error analysis for a specific problem shows O(h^2) error.

    Plot for heat flux balance for the whole fin.

"part_b.py"

    Plots temperature and heat flux over the fin, with a spatially varying fluid temp.

"part_c_rk2.py"

    Explicit RK2 method for the time-dependent problem. Makes contour plot for both values of omega, and also creates an animation.

"part_c_cn.py"

    Crank-Nicolson method for the time-dependent problem. Makes contour plot for both values of omega, and also creates an animation. The animations look a little rough since there aren't that many frames in the tau array.

"part_c_runtime.py"

    Runtime analysis for the RK4 time-dependent problem. Should only take a couple seconds to run at most.

"part_c_cn_long_prob.py"

    I wanted to just play with the solver, so I wrote an alternate fluid temperature function that generates a "step" function in the heat equation, just for fun.

===============================================

All the functions are in "toolbox". It has:

"differentiation.py" - integration/differentiation functions. Running this one directly just prints an array with all of the "intermediate" romberg integration values for the two methods for doing romberg integration; in one method, we have access to the function y(x) that's being integrated, in the other, we only have access to values of y(x) at regular grid points.

"formulation.py" - this one has the setup functions for the problems themselves.

    "const_step_grid" and "grid1d" -- function for generating grids

    "solve_ss_analytic" -- steady-state analytic solution for the ivp
    solve

    "get_A","get_b" -- standalone functions for getting A and b matrices

    "get_Ab" -- function to get both the A and b matrices for various fluid boundary conditions

    "gaussian_plume" -- function to generate a "gaussian" plume like that in problem b

    "solve_time_rk2" -- calculate explicit time-dependent solution

    "solve_time_crank" -- calculate crank-nicolson time-dependent solution
    
    "get_sin_plume_fn" and "get_step_plume_fn" -- functions for getting time+space dependent fluid functions. So you call this to get the function, and then call f(tau, xi) to get the value for each xi and tau.

    "animate_results" -- function to animate results from the time-dependent problems. You pass in length m tau, length n xi, and shape (m, n) fluid temp, shape (m,n) fin temp arrays, and it will make an animation of the fluid temp/fin temp over time

    "make_contour_plt" -- make a contour plot of a shape (m,n) temperature array

"solvers.py" - this one has the TDMA solver and a helper function for helping generate random tridiagonal matrices to test the solver.