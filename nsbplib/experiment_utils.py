import os
import numpy as np

def read_stats_file(stats_path):
    keywords = ['nfev','nit','njev','n_reg_jev']
    if not os.path.isfile(stats_path):
            raise RuntimeError('Cannot find stats: %s' % stats_path)
    with open(stats_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'n_reg_jev' in line:
                n_reg_jev = int(line.split(':')[1].strip())
            elif 'nfev' in line:
                n_fev = int(line.split(':')[1].strip())
            elif 'nit' in line:
                nit = int(line.split(':')[1].strip())
            elif 'njev' in line:
                njev = int(line.split(':')[1].strip())
            else:
                print('Skipping line...')
        return n_reg_jev,n_fev,nit,njev

def load_start_parameter(par,px=1,py=1):
    if '.npy' in str(par):
        x0 = np.load(par)
        p = int(np.sqrt(len(x0)))
        m = px // p
        x0 = x0.reshape((p,p))
        x0 = np.kron(x0,np.ones((m,m)))
        print(f'x0:{x0}')
    else:
        x0 = float(par)
        x0 = x0 * np.ones(px*py)
    return x0.ravel()

def save_nsbpl_results(evals,sol,extra_data,outfolder,run_name):
    print(f'{extra_data["noisy_imgs"].shape=}')
    num_training_samples = str(extra_data['true_imgs'].shape[2])
    # Exporting evals
    if not os.path.exists(os.path.join(outfolder,run_name,num_training_samples)):
        os.makedirs(os.path.join(outfolder,run_name,num_training_samples))
    evals_outfile = os.path.join(outfolder,run_name,num_training_samples,'%s_evals.pkl' % (run_name))
    evals.to_pickle(evals_outfile)
    print(f'Saved evals to: {evals_outfile}')
    
    if 'true_imgs' in extra_data:
        true_img_outfile = os.path.join(outfolder,run_name,num_training_samples, '%s_true_imgs.npy' % (run_name))
        np.save(true_img_outfile,extra_data['true_imgs'])
        print("Saved training data (true images) to: %s" % true_img_outfile)
    
    if 'noisy_imgs' in extra_data:
        true_img_outfile = os.path.join(outfolder,run_name,num_training_samples, '%s_noisy_imgs.npy' % (run_name))
        np.save(true_img_outfile,extra_data['noisy_imgs'])
        print("Saved training data (noisy images) to: %s" % true_img_outfile)
    
    if 'recons' in extra_data:
        true_img_outfile = os.path.join(outfolder,run_name,num_training_samples, '%s_recons.npy' % (run_name))
        np.save(true_img_outfile,extra_data['recons'])
        print("Saved training data (final reconstruction) to: %s" % true_img_outfile)
        
    # Write final statistics
    stats_outfile = os.path.join(outfolder,run_name,num_training_samples, '%s_stats.txt' % (run_name))
    with open(stats_outfile,'w') as f:
        print(sol,file=f)
    print("Saved experiment statistics to: %s" % stats_outfile)
        
    # Write the parameter
    opt_parameter_outfile = os.path.join(outfolder,run_name,num_training_samples,'%s_optimal_par.npy' % (run_name))
    np.save(opt_parameter_outfile,sol.x)
    print("Saved optimal parameter to: %s" % opt_parameter_outfile)