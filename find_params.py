import argparse
import mnist_cvae_train
from distutils.util import strtobool
from matplotlib import pyplot as plt
import os
from math import floor, ceil, log10

#import random # for debug

def plot(Ls, LDs, Ks, Cs, lambs, lambDs, prefix):
    """
    Helper function to produce plots from results of the hyperparameter search.
    They are saved in the results/parameter_search folder.
    Inputs:
        Ls, LDs - values for the first plot
        Ks, Cs - values for the second plot
        lambss, lambDs - values for the third plot
        prefix - filename prefix
    """
    
    # Create folder for results
    log_dir = os.path.join("results", "parameter_search")
    os.makedirs(log_dir, exist_ok=True)
    prefix = os.path.join(log_dir, prefix)
    
    # Plot D as a function of K+L
    plt.figure(figsize=(5,5))
    plt.plot(Ls, LDs)
    plt.scatter(Ls, LDs)
    plt.xticks(Ls, Ls)
    plt.xlabel(r"$K+L$")
    plt.ylabel(r"$\mathcal{D}$")
    plt.title("Step 1: select latent space dimension")
    plt.grid(alpha=0.5)
    plt.savefig(prefix + "_L.png")
    
    # Plot C as a function of K
    plt.figure(figsize=(5,5))
    plt.plot(Ks, Cs)
    plt.scatter(Ks, Cs)
    plt.ylim(floor(min(Cs) * 20) / 20, ceil(max(Cs) * 20) / 20)
    plt.xticks(Ks, Ks)
    plt.xlabel(r"$K$")
    plt.ylabel(r"$\mathcal{C}$")
    plt.title("Steps 2-3: increment causal factors")
    plt.grid(alpha=0.5)
    plt.savefig(prefix + "_K.png")
    
    # Plot D as a function of lambda
    plt.figure(figsize=(5,5))
    plt.plot(lambs, lambDs)
    plt.scatter(lambs, lambDs)
    plt.xscale('log')
    lps = [round(log10(l) * 2) / 2 for l in lambs]
    #plt.xticks(lambs, [r"$10^{{{}}}$".format(str(lp)) for lp in lps])
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\mathcal{D}$")
    plt.title(r"Steps 2-3: adjust $\lambda$")
    plt.grid(alpha=0.5, axis='x', which='both')
    plt.grid(alpha=0.5, axis='y')
    plt.savefig(prefix + "_lambda.png")



def find_params(args):
    """
    Function to find optimal values for K, L and lambda based on Algorithm 1.
    Produces plots that are saved in the results/parameter_search folder.
    Inputs:
        args - parse arguments containing other hyperparameters to use
    """
    
    print("Parameters:\n" + ", ".join([k + " = " + str(vars(args)[k]) for k in vars(args)]))
    
    print("\nFinding ideal number of latent dims...\n")
    
    # Initialize arguments such that only D is optimized
    args.lamb = 1.0
    args.use_C = False
    args.K = 0
    args.L = 0
    D_best = float('inf')
    Ls = []
    LDs = []
    
    # Increase L until D stops improving
    while True:
        args.L += 1
        result, _ = mnist_cvae_train.train(args)
        D_current = result[0]['Test ELBO']
        #D_current = random.randrange(100)
        print(f"K+L = {args.L}: D = {D_current:7.3f}")
        Ls.append(args.L)
        LDs.append(D_current)
        
        # Stop if D is worse than last D
        if D_current > D_best:
            # Continue using last value of L
            args.L -= 1
            print(f"Using {args.L} latent dims (D = {D_best:.3f}).\n")
            break
        
        # D has improved, so continue
        D_best = D_current
    
    print("Finding ideal number of causal factors...\n")
    
    # Now optimize D and C
    args.use_C = True
    C_best = float('-inf')
    lambda_best = None
    Ks = []
    Cs = []
    lambs = []
    lambDs = []
    
    # Make factors causal until C stops improving
    while args.L > 0:
        args.L -= 1
        args.K += 1
        lp = args.lambda_exp_0 - args.lambda_exp_step
        dist = float('inf')
        C_current = float('-inf')
        lambs_current = []
        lambDs_current = []
        
        print(f"Trying K = {args.K}, L = {args.L}:")
        
        # Increase lambda until D is close to D from above
        while lp <= 0:
            lp += args.lambda_exp_step
            args.lamb = pow(10, lp)
            result, _ = mnist_cvae_train.train(args)
            D_current = result[0]['Test ELBO']
            C_new = result[0]['Test Information Flow']
            #D_current = random.randrange(100)
            #C_new = random.uniform(-1, 0)
            print(f"lambda = {args.lamb:7.5f}: D = {D_current:7.3f}, C = {C_new:6.3f}")
            lambs_current.append(args.lamb)
            lambDs_current.append(D_current)
            dist_new = abs(D_current - D_best)
            
            # Stop if D is close enough to target D
            if dist_new < D_best * args.epsilon:
                # In this case, use current values of lambda and C
                C_current = C_new
                print(f"Using K = {args.K}, L = {args.L}, lambda = {args.lamb:.5f}, C = {C_current:.3f}.\n")
                break
            # Also stop if D is further away from target than last D
            elif dist_new > dist:
                # In this case, use last values of lambda and C
                args.lamb = pow(10, lp - args.lambda_exp_step)
                print(f"Using K = {args.K}, L = {args.L}, lambda = {args.lamb:.5f}, C = {C_current:.3f}.\n")
                break
            
            # D has improved, so continue
            C_current = C_new
            dist = dist_new
        
        Ks.append(args.K)
        Cs.append(C_current)
        
        # Stop if C is worse than last C
        if C_current < C_best:
            # Use best parameters found
            args.L += 1
            args.K -= 1
            args.lamb = lambda_best
            print("Final parameters:")
            print(f"K = {args.K}, L = {args.L}, lambda = {args.lamb:.5f}")
            break
        
        # C has improved, so continue
        C_best = C_current
        lambda_best = args.lamb
        lambs = lambs_current
        lambDs = lambDs_current
    
    if args.K == 0 or args.L == 0:
        print("Could not find good configuration.")
    
    # Save results
    class_str = ''.join(str(x) for x in sorted(args.classes))
    plot(Ls, LDs, Ks, Cs, lambs, lambDs, class_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Search parameters
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='Leeway factor to decide when D is close enough to original D')
    parser.add_argument('--lambda_exp_0', default=-3, type=float,
                        help='Initial exponent for lambda factor. First lambda value will be 10^(lambda_exp_0)')
    parser.add_argument('--lambda_exp_step', default=0.5, type=float,
                        help='Value by which to increase lambda exponent each step. \
                            At a value of 1, lambda will be increased by a factor of 10 each step.')
    
    # Model hyperparameters
    parser.add_argument('--classes', default=[3, 8],
                        type=int, nargs='+',
                        help='The classes permittible for classification')
    parser.add_argument('--classifier_path', type=str, 
                        help='This is the directory INSIDE of models where pre-trained \
                            black-box classifier is. Necessary if naming convention is not \
                                adhered to')
    parser.add_argument('--num_filters', default=64, type=int,
                        help='Number of filters used in the encoders/decoders')
    parser.add_argument('--M', default=2, type=int,
                        help='Dimensionality of classifier output')
    
    # Loss and optimizer hyperparameters
    parser.add_argument('--max_steps', default=8000, type=int,
                        help='Max number of training batches')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--Nalpha', default=100, type=int,
                        help='Learning rate to use')
    parser.add_argument('--Nbeta', default=25, type=int,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--betas', default=[0.5, 0.99],
                        type=int, nargs=2,
                        help='The beta parameters for add_argument')
    
    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--progress_bar', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--sample_every', default=-1, type=int,
                        help='When to sample the latent space. If -1, only samples at end of training.')
    parser.add_argument('--log_dir', default='mnist_cvae', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--add_classes_to_cpt_path', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether to add the classes to cpt directory.')
    parser.add_argument('--silent', default=True, type=lambda x: bool(strtobool(x)),
                        help='Perform training without printing to console or creating graphs.')

    # Debug parameters
    parser.add_argument('--debug', default=False, type=lambda x: bool(strtobool(x)),
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to train on GPU (if available) or CPU'))
    parser.add_argument('--num_workers', default=0, type=int,
                        help=('Number of workers to use for the dataloaders.'))
    
    args = parser.parse_args()

    find_params(args)