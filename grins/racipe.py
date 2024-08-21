import os
import glob
import argparse
import pandas as pd
from grins.generate_params import parse_topos, gen_param_names, gen_param_df, get_param_range_df
from grins.gen_diffrax_ode import gen_diffrax_odesys

def main():
    parser = argparse.ArgumentParser(\
        prog='RACIPE',
        description='Run simulation of GRN-ODE model for ensemble of parameters')
    parser.add_argument('topo', type=str, help='topo file name', default='all', nargs='?')
    parser.add_argument('--topodir', type=str, help='topo file directory', default='TOPOS')
    parser.add_argument('--simdir', type=str, help='simulation directory', default='SimResults')
    parser.add_argument('--numparas', type=int, help='number of parameters', default=10000)
    args = parser.parse_args()
    # if no topo file is provided, use iterate over all the topo files
    topos = sorted(glob.glob(f"{args.topodir}/*.topo")) if args.topo == 'all' else [args.topo]
    for tpfl in topos:
        # Get the topo name
        topo_name = tpfl.split("/")[-1].split(".")[0]
        print(f'Generating ODE for {topo_name}')
        # Create the directory
        os.makedirs(f"{args.simdir}/{topo_name}", exist_ok=True)
        # Parse the topology file
        topo_df = parse_topos(tpfl)
        # Generate the parameter names
        prange_df = get_param_range_df(topo_df)
        prange_df.to_csv(f"{args.simdir}/{topo_name}/{topo_name}.prs", index=False, sep=' ')
        # Generate the parameter dataframe
        param_df = gen_param_df(prange_df, args.numparas)
        param_df.to_csv(f"{args.simdir}/{topo_name}/{topo_name}_params.dat", index=False, sep=' ')
        # Generate the diffrax ode system
        gen_diffrax_odesys(
            topo_df, tpfl, save_dir=f"{args.simdir}/{tpfl.split('/')[-1].split('.')[0]}"
        )


if __name__ == '__main__':
    main()