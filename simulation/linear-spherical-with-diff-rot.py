# 1D MHD Rossby wave equations in spherical coordinates with differential rotation
# Based on equations (10)-(11) from Zaqarashvili et al. 2010
# Simulates in mu = cos(theta) with fixed azimuthal wavenumber m
# Start with: python3 linear-spherical-with-diff-rot.py input_spherical.json
# or for multi-processor: mpiexec -n 4 python3 linear-spherical-with-diff-rot.py input_spherical.json

import json
import sys
import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI

comm = MPI.COMM_WORLD

handlers = [logging.StreamHandler(sys.stdout)]

# --------- SIMPLE LOGGING: terminal + file ----------
if comm.rank == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s :: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logger_info.txt", mode="w"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    solvers_logger = logging.getLogger("solvers")
    solvers_logger.setLevel(logging.INFO)
    solvers_logger.propagate = False
    for h in logging.getLogger().handlers:
        if h not in solvers_logger.handlers:
            solvers_logger.addHandler(h)
else:
    logging.basicConfig(level=logging.WARNING, force=True)
    logger = logging.getLogger(__name__)
# ---------------------------------------------------

restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
if len(sys.argv)==1:
   print('Argument missing!/Argument fehlt!')
   exit()
else:
   params=json.load(open(sys.argv[1]))

# Rossby wave parameters
m = params['m']  # azimuthal wavenumber
s2 = params['s2']  # differential rotation parameter
s4 = params['s4']  # differential rotation parameter
beta_sq = params['beta_sq']  # magnetic parameter β²
Nmu = params['Nmu']  # resolution in mu direction
R = params['R']

# Initial condition parameters
IC_amp = params["IC_amp"]
mu_center = params["mu_center"]
mu_width = params["mu_width"]

stop_sim_time = params['stop_sim_time'] + params['stop_sim_time']*restart
max_timestep = params['max_timestep']

run_name = params['run_name']
ic_name = params['ic_name']

snapshots_dir = run_name+'_snapshots'
state_dir = run_name+'_state'

dealias = 3/2
timestepper = d3.SBDF2
dtype = np.complex128  # Complex fields for wave dynamics

nproc = comm.Get_size()
mesh = None

# Bases
coords = d3.CartesianCoordinates('mu')
#dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
dist = d3.Distributor(coords, dtype=dtype)
mu_basis = d3.Chebyshev(coords['mu'], size=Nmu, bounds=(-1, 1), dealias=dealias)

# Fields (all complex)
Psi = dist.Field(name='Psi', bases=mu_basis)  # Stream function
Phi = dist.Field(name='Phi', bases=mu_basis)  # Magnetic stream function
varphi = dist.Field(name='varphi', bases=mu_basis)  # Auxiliary: varphi = L(Psi)

# Substitutions
mu_grid = dist.local_grid(mu_basis)

# Create mu as a Field for use in equations
mu = dist.Field(name='mu', bases=mu_basis)
mu['g'] = mu_grid

# Create (1 - mu^2) as a Field for convenience
one_minus_mu2 = dist.Field(name='one_minus_mu2', bases=mu_basis)
one_minus_mu2['g'] = 1 - mu_grid**2

# Differential rotation profile: Omega_d = -s2*mu^2 - s4*mu^4
Omega_d = dist.Field(name='Omega_d', bases=mu_basis)
Omega_d['g'] = -s2 * mu_grid**2 - s4 * mu_grid**4

mu2_term = dist.Field(name='mu2_term', bases=mu_basis)
mu2_term['g'] = s2 * mu_grid**2 * R

mu4_term = dist.Field(name='mu4_term', bases=mu_basis)
mu4_term['g'] = s4 * mu_grid**4 * R


# Magnetic field profile: B = mu
B = dist.Field(name='B', bases=mu_basis)
#B['g'] = mu_grid
B['g'] = 0

# Derivatives
dmu = lambda A: d3.Differentiate(A, coords['mu'])

# Legendre operator: L = d/dmu[(1-mu^2)d/dmu] - m^2/(1-mu^2)
def Legendre_op(A):
    """Apply Legendre operator to field A"""
    return one_minus_mu2 * dmu(dmu(A)) - 2*mu*dmu(A) - m**2 * A / one_minus_mu2

# Second derivative term: d^2/dmu^2[f(mu)(1-mu^2)]
def d2_term(f):
    """Compute d^2/dmu^2[f(1-mu^2)]"""
    # f(1-mu^2) where f is a field
    temp = f * one_minus_mu2
    return dmu(dmu(temp))

# --- tau/lift setup (Chebyshev) ---
lift_basis = mu_basis.derivative_basis(1)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# tau fields for boundary conditions
tau_Psi1 = dist.Field(name='tau_Psi1')
tau_Psi2 = dist.Field(name='tau_Psi2')
tau_Phi1 = dist.Field(name='tau_Phi1')
tau_Phi2 = dist.Field(name='tau_Phi2')

# --- Problem ---
problem = d3.IVP([Psi, varphi, tau_Psi1, tau_Psi2], namespace=locals())

# Equation for varphi = L(Psi) evolution (from equation 10)
# i*dt(varphi) + Omega_d*varphi + [2 - d²/dμ²[Ωd(1-μ²)]]Ψ - β²*B*L(Φ) + β²*d²/dμ²[B(1-μ²)]Φ = 0

problem.add_equation("1j*dt(varphi) - Omega_d*varphi - (2 - d2_term(Omega_d))*Psi = 0")

# Equation for Phi evolution (from equation 11)
# i*dt(Phi) + Omega_d*Phi - B*Psi = 0
#problem.add_equation("1j*dt(Phi) - Omega_d*Phi + B*Psi + lift(tau_Phi1, -1) + lift(tau_Phi2, -2) = 0")

# Diagnostic equation: L(Psi) = varphi
problem.add_equation("Legendre_op(Psi) - varphi + lift(tau_Psi1, -1) + lift(tau_Psi2, -2) = 0")

# Boundary conditions: Psi = Phi = 0 at poles (mu = ±1)
problem.add_equation("Psi(mu=-1) = 0")
problem.add_equation("Psi(mu=1) = 0")
#problem.add_equation("Phi(mu=-1) = 0")
#problem.add_equation("Phi(mu=1) = 0")
'''
# Additional BC for varphi (consistency)
problem.add_equation("varphi(mu=-1) = 0")
problem.add_equation("varphi(mu=1) = 0")
'''

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if ic_name == 'none':
    '''
    # Legendre polynomial initial condition
    # Use associated Legendre polynomial P_l^m(mu) for initial perturbation
    # l = degree (latitudinal mode), m = order (azimuthal mode)
    # Constraint: l >= |m|
    
    from scipy.special import lpmv
    
    # Get l mode from parameters (default to l=m+1 if not specified)
    l_mode = params.get('n', m + 1)
    
    if l_mode < abs(m):
        logger.error(f'Invalid mode selection: l={l_mode} must be >= |m|={abs(m)}')
        raise ValueError(f'Legendre polynomial P_{l_mode}^{m} does not exist (requires l >= |m|)')
    
    # Calculate associated Legendre polynomial P_l^m(mu)
    # scipy.special.lpmv(m, l, mu) computes P_l^m(mu)
    # Note: scipy uses a different normalization than some references
    Psi_initial = lpmv(m, l_mode, mu_grid)
    
    # Normalize to have maximum absolute value of IC_amp
    max_val = np.max(np.abs(Psi_initial))
    if max_val > 0:
        Psi['g'] = IC_amp * Psi_initial / max_val 
        #IC_amp controls the maximum amplitude of the spatial pattern, not a constant value
    else:
        logger.error(f'P_{l_mode}^{m}(mu) is zero everywhere')
        raise ValueError(f'Invalid Legendre polynomial P_{l_mode}^{m}')
    
    logger.info(f'Initial condition: Associated Legendre polynomial P_{l_mode}^{m}(mu) with amplitude={IC_amp}')
    logger.info(f'  Number of nodes between poles: {l_mode - m}')
    '''
    #Gaussian in Psi
    Psi['g'] = IC_amp * np.exp(-((mu_grid - mu_center)**2) / mu_width**2)
    # Phi starts at zero
    Phi['g'] = 0
    
    # Compute initial varphi = L(Psi)
    d_Psi = np.gradient(Psi['g'], mu_grid) 
    d2_Psi = np.gradient(d_Psi, mu_grid) 
    varphi['g'] = (1 - mu_grid**2) * d2_Psi - 2*mu_grid * d_Psi - m**2 * Psi['g'] / (1 - mu_grid**2 + 1e-10)
else:
    ic_file = ic_name + '_state/' + ic_name + '_state_s1.h5'
    write, initial_time = solver.load_state(ic_file)
    logger.info('Starting at t = %f', solver.sim_time)
    stop_sim_time += solver.sim_time
    solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=500)
snapshots.add_task(Psi, name='Psi')
snapshots.add_task(Phi, name='Phi')
snapshots.add_task(varphi, name='varphi')
snapshots.add_task(np.real(Psi), name='Psi_real')
snapshots.add_task(np.imag(Psi), name='Psi_imag')
snapshots.add_task(np.real(Phi), name='Phi_real')
snapshots.add_task(np.imag(Phi), name='Phi_imag')
snapshots.add_task(np.abs(Psi), name='Psi_abs')
snapshots.add_task(np.abs(Phi), name='Phi_abs')

# Add velocity components from equation (7)
# u_theta = (im*Psi) / (R*sin(theta)) = (im*Psi) / (R*sqrt(1-mu^2))
# u_phi =  d(Psi)/dtheta =  d(Psi)/dmu * dmu/dtheta =  d(Psi)/dmu / sin(theta)
# For non-dimensionalized versions (R=1):
# u_theta = (im*Psi) / sqrt(1-mu^2)
# u_phi = d(Psi)/dmu / sqrt(1-mu^2)

u_theta = (1j*m*Psi) / (one_minus_mu2**0.5 + 1e-10)
u_phi = dmu(Psi) * (one_minus_mu2**0.5) + ( - mu2_term - mu4_term) * (one_minus_mu2**0.5)

snapshots.add_task(u_theta, name='u_theta')
snapshots.add_task(u_phi, name='u_phi')
snapshots.add_task(np.real(u_theta), name='u_theta_real')
snapshots.add_task(np.imag(u_theta), name='u_theta_imag')
snapshots.add_task(np.abs(u_theta), name='u_theta_abs')
snapshots.add_task(np.real(u_phi), name='u_phi_real')
snapshots.add_task(np.imag(u_phi), name='u_phi_imag')
snapshots.add_task(np.abs(u_phi), name='u_phi_abs')

# Add magnetic field components from equation (7)
# b_theta = (im*Phi) / (R*sin(theta)) = (im*Phi) / sqrt(1-mu^2)
# b_phi =  d(Phi)/dtheta = d(Phi)/dmu / sqrt(1-mu^2)

b_theta = (1j*m*Phi) / (one_minus_mu2**0.5 + 1e-10)
b_phi = dmu(Phi) * (one_minus_mu2**0.5)

snapshots.add_task(b_theta, name='b_theta')
snapshots.add_task(b_phi, name='b_phi')
snapshots.add_task(np.real(b_theta), name='b_theta_real')
snapshots.add_task(np.imag(b_theta), name='b_theta_imag')
snapshots.add_task(np.abs(b_theta), name='b_theta_abs')
snapshots.add_task(np.real(b_phi), name='b_phi_real')
snapshots.add_task(np.imag(b_phi), name='b_phi_imag')
snapshots.add_task(np.abs(b_phi), name='b_phi_abs')

# Save final state
final_state = solver.evaluator.add_file_handler(state_dir, sim_dt=stop_sim_time-max_timestep, mode='overwrite')
final_state.add_tasks(solver.state)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05, max_dt=max_timestep)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.abs(Psi), name='Psi_abs')
flow.add_property(d3.abs(Phi), name='Phi_abs')

# Main loop
try:
    logger.info('Starting main loop')
    logger.info('Parameters: m=%d, s2=%.4f, s4=%.4f, beta_sq=%.6f', m, s2, s4, beta_sq)
    if 'n_mode' in params:
        logger.info('Initial condition: Legendre polynomial P_%d^%d(mu) with amplitude=%.4f', 
                    params.get('n_mode', 2), m, IC_amp)
    else:
        logger.info('Initial condition: Gaussian with amplitude=%.4f, center=%.2f, width=%.2f', 
                    IC_amp, mu_center, mu_width)
    while solver.proceed:
        #timestep = CFL.compute_timestep()
        timestep = max_timestep
        solver.step(timestep)
        if (solver.iteration) % 100 == 0:
            max_Psi = flow.max('Psi_abs')
            max_Phi = flow.max('Phi_abs')
            logger.info('Iteration=%i, Time=%.4f, dt=%e, max(|Psi|)=%e, max(|Phi|)=%e' 
                       %(solver.iteration, solver.sim_time, timestep, max_Psi, max_Phi))
        
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

logger.info('Simulation completed successfully')