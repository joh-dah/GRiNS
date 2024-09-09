import os
import shutil
import glob
import argparse
import pandas as pd
from grins.generate_params import parse_topos, gen_param_df, get_param_range_df, gen_init_cond, get_thr_ranges
from grins.gen_diffrax_ode import gen_diffrax_odesys
from grins.sim_run import get_steady_states, ODE_topo

def main():
    parser = argparse.ArgumentParser(\
        prog='RACIPE',
        description='Run simulation of GRN-ODE model for ensemble of parameters')
    parser.add_argument('topo', type=str, help='topo file name', default='all', nargs='?')
    parser.add_argument('--topodir', type=str, help='topo file directory', default='TOPOS')
    parser.add_argument('--outdir', type=str, help='simulation directory', default='SimResults')
    parser.add_argument('--num_paras', type=int, help='number of parameters', default=10000)
    parser.add_argument('--num_inits', type=int, help='number of initial conditions', default=1000)
    parser.add_argument('--num_reps', type=int, help='number of replicates', default=3)
    parser.add_argument('--num_cores', type=int, help='number of cores', default=0)
    parser.add_argument('--genprs', action='store_true', help='generate parameter range file, no simulation')
    parser.add_argument('--prsfile', type=str, help='parameter range file name', default=None)
    parser.add_argument('--no_calc_thrs', action='store_true', help='Do not calculate thresholds, used with prsfile')
    parser.add_argument('--sampling', type=str, help='sampling method. Choices: sobol, uni, loguni, latin_hc', default='sobol')
    # parser.add_argument('--gencfg', type=bool, help='generate config file only, no simulation', default=False)
    # parser.add_argument('--cfgfile', type=str, help='config file name', default=None)
    args = parser.parse_args()
    # if no topo file is provided, use iterate over all the topo files
    topos = sorted(glob.glob(f"{args.topodir}/*.topo")) if args.topo == 'all' else [args.topo]
    for tpfl in topos:
        # Get the topo name
        topo_name = tpfl.split("/")[-1].split(".")[0]
        # Create the directory
        os.makedirs(f"{args.outdir}/{topo_name}", exist_ok=True)
        # Parse the topology file
        topo_df = parse_topos(tpfl)
        # Copy topofile to outdir
        shutil.copy(tpfl, f"{args.outdir}/{topo_name}")
        # Switch cases for cfg, prs
        if args.genprs:
            gen_prs(topo_name, topo_df, args)
        else:
            default(topo_name, topo_df, args)

def gen_prs(topo_name, topo_df, args):
    print(f'Generating Parameter range for {topo_name}')
    # Generate the parameter names
    prange_df = get_param_range_df(topo_df, sampling=args.sampling)
    prange_df.to_csv(f"{args.outdir}/{topo_name}/{topo_name}.prs", index=True, sep='\t')
    return prange_df
    
def default(topo_name, topo_df, args):
    # Generate the parameter names
    if args.prsfile:
        prange_df = pd.read_csv(args.prsfile, sep='\t', index_col=0)
        if not args.no_calc_thrs:
            get_thr_ranges(topo_df, prange_df)
    else:
        prange_df = gen_prs(topo_name, topo_df, args)
    # Generating parameters for the topology
    # Generate the parameter dataframe
    print(f'Generating Parameters for {topo_name}')
    param_df = gen_param_df(prange_df, num_paras=args.num_paras, sampling=args.sampling)
    param_df.to_csv(f"{args.outdir}/{topo_name}/{topo_name}_params.dat", sep='\t', index=True)
    # Generate the initial conditions dataframe
    init_df = gen_init_cond(topo_df, num_init_conds=args.num_inits, sampling=args.sampling)
    init_df.to_csv(f"{args.outdir}/{topo_name}/{topo_name}_init_conds.csv", index=True)
    print(f'Generating ODE for {topo_name}')
    # Generate the diffrax ode system
    gen_diffrax_odesys(
        topo_df, topo_name, save_dir=f"{args.outdir}/{topo_name}"
    )
    term = ODE_topo(topo_name, args.outdir)
    # Get the steady states
    print(f'Calculating steady states for {topo_name}')
    sol_df = get_steady_states(init_df, param_df, term)
    sol_df.to_parquet(f"{args.outdir}/{topo_name}/{topo_name}_sol.parquet", index=True)

if __name__ == '__main__':
    main()