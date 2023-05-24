
import numpy as np
import pandas as pd
from nsbplib.data_generation import load_training_data
from nsbplib.lower_level_problems import get_lower_level_problem


def _to_array(x,lbl):
    try:
        if isinstance(x,float):
            x = [x]
        return np.asarray_chkfinite(x)
    except ValueError:
        raise ValueError('%s contains Nan/Inf values' % lbl)

class UpperLevelProblem(object):
    def __init__(self,lower_level_problems,true_imgs):
        self.lower_level_problems = lower_level_problems
        self.true_imgs = true_imgs
        self.nsamples = len(true_imgs)
        
        # Store Upper Level Problem Evaluation History
        self.costs = []
        self.grads = []
        
    def get_resid(self,param,smooth):
        resid = np.zeros((self.nsamples,))
        grads = np.zeros((self.nsamples,len(param)))
            
        for i in range(self.nsamples):
            self.lower_level_problems[i](param)
            resid[i] = self.lower_level_problems[i].loss(self.true_imgs[i])
            if smooth:
                grads[i,:] = self.lower_level_problems[i].smooth_grad(self.true_imgs[i],param)
            else:
                grads[i,:] = self.lower_level_problems[i].grad(self.true_imgs[i],param)
        return resid,grads
    
    def __call__(self, param, smooth=False):
        resids,grads = self.get_resid(param,smooth)
        cost = np.sum(resids) / self.nsamples
        grad = np.sum(grads,axis=0) / self.nsamples
        # print(cost,grad)
        self.costs.append(cost)
        self.grads.append(grad)
        # print(f'grads:\n{grads}')
        return cost,_to_array(grad,'g')
        
    def get_evals(self):
        mydict = {
            'eval':np.arange(len(self.costs)),
            'f':self.costs,
            'g':self.grads
        }
        return pd.DataFrame.from_dict(mydict)
    
    def get_training_data(self):
        # Get all training data in a easily-saved format
        if len(self.true_imgs[0].shape) == 1:  # 1D images
            true_imgs = np.vstack(self.true_imgs)  # each row is an image
            noisy_imgs = np.vstack([s.data for s in self.lower_level_problems])
            recons = np.vstack([s.recon for s in self.lower_level_problems])
        elif len(self.true_imgs[0].shape) == 2:  # 2D images
            true_imgs = np.dstack(self.true_imgs)  # true_imgs[:,:,i] is image i
            noisy_imgs = np.dstack([s.data for s in self.lower_level_problems])
            recons = np.dstack([s.recon for s in self.lower_level_problems])
        else:
            raise RuntimeError("Do not have ability to append 3D or above data yet")
        return true_imgs, noisy_imgs, recons
        
class UpperScalarDataLearning(UpperLevelProblem):
    def __init__(self, ds_dir, num_training_data:int=1, verbose = False):

        # Define the problem type
        problem_type='2D_scalar_data_learning'
        
        # Build Training Data
        num_training_data, true_imgs, noisy_imgs = load_training_data(ds_dir,num_training_data=num_training_data)
        lower_level_problems = []
        
        for i in range(num_training_data):
            # true_img = true_imgs[i]
            noisy_img = noisy_imgs[i]
            llproblem = get_lower_level_problem(problem_type, noisy_img, "Img %g" % i,1,1)
            lower_level_problems.append(llproblem)
            
        print(f'Starting optimal parameter learning with\ntraining set:{ds_dir}\nproblem_type:{problem_type}\nnum_training_data:{num_training_data}')
        
        super().__init__(lower_level_problems, true_imgs)
        
class UpperPatchDataLearning(UpperLevelProblem):
    def __init__(self, ds_dir,num_training_data, px, py, verbose = False):

        # Define the problem type
        problem_type='2D_patch_data_learning'
        
        # Build Training Data
        num_training_data, true_imgs, noisy_imgs = load_training_data(ds_dir,num_training_data=num_training_data)
        lower_level_problems = []
        
        for i in range(num_training_data):
            noisy_img = noisy_imgs[i]
            llproblem = get_lower_level_problem(problem_type, noisy_img, "Img %g" % i,px,py)
            lower_level_problems.append(llproblem)
            
        print(f'Starting optimal data parameter learning with\ntraining set:{ds_dir}\nproblem_type:{problem_type}\nnum_training_data:{num_training_data}\npatch_size:{px}x{py}')
        
        super().__init__(lower_level_problems, true_imgs)
        
class UpperScalarRegLearning(UpperLevelProblem):
    def __init__(self, ds_dir, seed = 0, verbose = False):

        # Define the problem type
        problem_type='2D_scalar_reg_learning'
        
        # Build Training Data
        num_training_data, true_imgs, noisy_imgs = load_training_data(ds_dir)
        lower_level_problems = []
        
        for i in range(num_training_data):
            # true_img = true_imgs[i]
            noisy_img = noisy_imgs[i]
            # noisy_img = true_img + noise_level * np.random.randn(*true_img.shape)
            llproblem = get_lower_level_problem(problem_type, noisy_img, "Img %g" % i,1,1)
            lower_level_problems.append(llproblem)
            
        print(f'Starting optimal regularization parameter learning with\ntraining set:{ds_dir}\nproblem_type:{problem_type}\nnum_training_data:{num_training_data}')
        
        super().__init__(lower_level_problems, true_imgs)
        
class UpperPatchRegLearning(UpperLevelProblem):
    def __init__(self, ds_dir, px, py, seed = 0, verbose = False):

        # Define the problem type
        problem_type='2D_patch_reg_learning'
        
        # Build Training Data
        num_training_data, true_imgs, noisy_imgs = load_training_data(ds_dir)
        lower_level_problems = []
        
        # Fix random seed
        # np.random.seed(seed)
        
        for i in range(num_training_data):
            noisy_img = noisy_imgs[i]
            llproblem = get_lower_level_problem(problem_type, noisy_img, "Img %g" % i,px,py)
            lower_level_problems.append(llproblem)
            
        print(f'Starting optimal reg parameter learning with\ntraining set:{ds_dir}\nproblem_type:{problem_type}\nnum_training_data:{num_training_data}\npatch_size:{px}x{py}')
        
        super().__init__(lower_level_problems, true_imgs)