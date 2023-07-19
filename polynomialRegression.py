# load libraries
#================================================================================================================
# M
import matplotlib as mpl
import matplotlib.pyplot as plt
# N
import numpy as np
# S
import seaborn as sns # for the heatmap plot
# T
import torch
import torch.nn as nn 
import torch.optim as optim


#================================================================================================================
# custom-made functions
#================================================================================================================
# used at task
def plot_polynomial(coeffs, z_range , color='b'):
    ''' coeffs : tuple or list containing coefficients of 
        a polynomial in the increasing order of degrees 
        z_range: tuple or list of two elements
        representing the range of z (min, max)'''

    z = np.linspace(z_range[0], z_range[1], 100)
    y = np.polynomial.polynomial.polyval(z, coeffs)
    plt.plot(z, y, color)
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    plt.title("Polynomial graph\n$p(x) = 0.05x^4 + x^3 + 2x^2 - 5x$")
    plt.savefig('fig1-polynomial.png', dpi=300)

# used at task
def create_dataset(w, z_range , sample_size , sigma, seed=42):
    ''' w: list(int), parameters to be estimated later
        z_range: 2tuple(float), range min max of z values
        sample_size: int, number of elements in the dataset
        sigma: float > 0, noise of the dataset w.r.t function
        seed: for random state'''

    random_state = np.random.RandomState(seed)
    z = random_state.uniform(z_range[0], z_range[1], (sample_size)) 
    
    x = np.zeros((sample_size , w.shape[0]))
    
    for i in range(sample_size):
        for j in range(len(w)):
            x[i,j] = z[i] ** j
        
    y = x.dot(w) 
        
    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size ) 
        
    return x, y

# plotting function
def plot_datasets(dataset_list, output_name, type_of_plot, poly=None):

    if poly is not None:
        p_labels, zs, ws = poly

    fig = plt.figure()
    fig.set_size_inches(10, 4)

    if dataset_list is None:
        dataset_list = [(None, None, None)]
        
    for i, dataset in enumerate(dataset_list):
        label, xlab, x, ylab, y = dataset

        plt.subplot(1, 2, i + 1)
        if poly is not None: 
            p_of_z = np.polynomial.polynomial.polyval(zs, ws)
            plt.plot(zs, p_of_z, color='crimson', label=p_labels, alpha=0.6)
        
        if type_of_plot == "plot":
            plt.plot(x, y, color='steelblue', label=label)
                
        elif type_of_plot == "scatter":
            plt.scatter(x, y, color='steelblue', marker='o', s=1, label=label)

        plt.title(label)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)   

# training function
def fit_polynomial(trn_data, val_data, hyp_parm, benchmark_mode=False, study_weights=False):
    
    # unpacking assets
    #============================================================================
    X_trn, y_trn = trn_data
    X_val, y_val = val_data
    alpha, nstep = hyp_parm
    if nstep == 0: nstep = 1

    # setting the model
    #============================================================================
    # 1. Linear Model
    
    model = nn.Linear(in_features=5, out_features=1, bias=False, device=DEVICE)
    # bias is turned off so we include the zero-th grade coeffitient in weight list

    # if not benchmark_mode: print(f'Initializing {model.weight}\n#============================================================================')

    # 2. Loss function (MSE)
    loss_fn = nn.MSELoss()

    # array for storing later training / evaluation performance
    trn_loss_list = np.zeros(nstep)
    val_loss_list = np.zeros(nstep)
    par_evolution = np.zeros((nstep, 5))

    # 3. Gradient calculator
    optimizer = optim.SGD(model.parameters(), lr=alpha)

    for step in range(nstep):
    
        model.train()
        #============================================================================
        # 1. prediction of the model
        y_ = model(X_trn)
        
        # 2. loss of the predicted y_ wrt the target y
        trn_loss = loss_fn(y_, y_trn)
        
        # 3. compute gradients
        trn_loss.backward()
        
        # 4. update estimated parameters using gradients
        optimizer.step()
        
        # 5. reset gradients to 0
        optimizer.zero_grad()

        # 6. save training loss for performance assessment
        trn_loss_list[step] = trn_loss


        model.eval()
        #============================================================================
        with torch.no_grad():
            # 1. compute estimations of the target in the val dataset
            y_ = model(X_val)
            
            # 2. compute validation loss
            val_loss = loss_fn(y_, y_val)

            # 3. save validation loss for performance assessment
            val_loss_list[step] = val_loss

            if study_weights: par_evolution[step, :] = model.weight

    if not benchmark_mode:
        # print(f'Estimated {model.weight}\n#============================================================================')
        # print(f'Training loss: {trn_loss}\t| Validation loss: {val_loss}')

        # return the weights evolution if flag is true
        if study_weights: return par_evolution

        # plotting of the evolution of the losses with the iterations
        #============================================================================
        plot_datasets([("Training Loss", "step", range(nstep), "Loss", trn_loss_list),
                       ("Validation Loss", "step", range(nstep), "Loss", val_loss_list)], 
                       output_name="fig3-losses.png",
                       type_of_plot="plot")

        # return the training model and the loss_lists
        return model, (trn_loss_list, val_loss_list)

    else:
        # return only the last losses
        return trn_loss, val_loss



def plot_heatmaps(dataset_list, ticks, output_name):
    alpha_fn, nstep_fn, num_tries, ylabs = ticks
    alpha_num_tries, nstep_num_tries = num_tries

    fig = plt.figure()
    fig.set_size_inches(20, 8)
    
    #print(alpha_fn([alpha_fn(j) for j in range(alpha_num_tries)]))

    for i, dataset in enumerate(dataset_list):
        label, xlab, ylab, matrix_record, loss_limit= dataset

        plt.subplot(1, 2, i + 1)
        sns.heatmap(matrix_record, 
                    # annot=True, fmt='.2g',
                    cbar_kws={"label": "Loss of the model"},
                    xticklabels=[nstep_fn(j) for j in range(nstep_num_tries)],
                    yticklabels=ylabs, 
                    linewidth=0.5, 
                    vmin=0, 
                    vmax=loss_limit)
        plt.title(label)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.show()  


# benchmark function
def benchmark_hyp_parm(alpha_fn, nstep_fn, num_tries, plot_loss_up_to, ylabs):
    alpha_num_tries, nstep_num_tries = num_tries

    # benchmark will store the losses for different combinations of hyperparameters
    trn_benchmark = np.zeros(shape=num_tries)
    val_benchmark = np.zeros(shape=num_tries)

    for k in range(alpha_num_tries):
        for j in range(nstep_num_tries):
            
            try_alpha = alpha_fn(k)
            try_nstep = nstep_fn(j)

            trn_loss, val_loss = fit_polynomial(trn_data=(X_trn, y_trn), 
                                                val_data=(X_val, y_val), 
                                                hyp_parm=(try_alpha, try_nstep),
                                                benchmark_mode=True)

            trn_benchmark[k, j] = trn_loss
            val_benchmark[k, j] = val_loss

    
    plot_heatmaps(dataset_list=[("Training Benchmark", "Number of Steps", "Learning rate", trn_benchmark, plot_loss_up_to),
                                ("Validation Benchmark", "Number of Steps", "Learning rate", val_benchmark, plot_loss_up_to)], 
                  ticks=(alpha_fn, nstep_fn, num_tries, ylabs), 
                  output_name="fig4-benchmark.png")

    return trn_benchmark, val_benchmark

# for task 11
def plot_weight_evolution(par_evolution):
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)

    for i in range(5):

        plt.subplot(2, 3, i + 1)
        plt.plot(range(nstep), par_evolution[:, i], color='steelblue', label=f'$w_{i}$ evolution throughout training process')
        plt.plot(range(nstep), np.full((nstep, ), parameter[i]), color='crimson', label=f'$w_{i}$ true value')
        plt.title(f'Evolution of $w_{i}$')
        plt.xlabel('step')
        plt.ylabel(f'$w_{i}$ value')
        plt.ylim(min(np.min(par_evolution[:, i]), parameter[i]) - 0.1, max(np.max(par_evolution[:, i]), parameter[i]) + 0.1)
        plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig("fig10-evolution-weights.png", dpi=300)
    plt.show()

#================================================================================================================
# task 1: Graph polynomial
#================================================================================================================
parameter = np.array([0, -5, 2, 1, 0.05]) 
plot_polynomial(coeffs=parameter, z_range=(-3, 3))


#================================================================================================================
# task 2: Polynomial regression dataset generation
#================================================================================================================
# done inside function create_dataset()


#================================================================================================================
# task 3: Generating training and validation datasets
#================================================================================================================
print("""starting task 3...
generating datasets""")

(raw_X_trn, raw_y_trn) = create_dataset(w=parameter, z_range=(-3, 3), sample_size=500, sigma=0.5, seed=0)
(raw_X_val, raw_y_val) = create_dataset(w=parameter, z_range=(-3, 3), sample_size=500, sigma=0.5, seed=1)

print("""datasets successfully generated
================================================================================================================""")


#================================================================================================================
# task 4: Visualize the generated training and validation data points
#================================================================================================================
print("""starting task 4...
plotting datasets datasets""")

truepoly = ("Generating Polynomial", np.linspace(-3, 3, 100), [0, -5, 2, 1, 0.05])
datasets = [("Training Dataset", "x", raw_X_trn[:,1], "y", raw_y_trn), 
            ("Validation Dataset", "x", raw_X_val[:,1], "y", raw_y_val)]

plot_datasets(datasets, "fig2-trn-val-sets.png", "scatter", truepoly)

print("""datasets successfully plotted
================================================================================================================""")


#================================================================================================================
# task 5: Bias flag in nn.Linear()
#================================================================================================================


#================================================================================================================
# task 6: Polynomial linear regression
#================================================================================================================
print("""starting task 6...
learning weights through training dataset""")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data preprocessing

X_trn = torch.from_numpy(raw_X_trn).float().to(DEVICE)
y_trn = torch.from_numpy(raw_y_trn).reshape(500, 1).float().to(DEVICE)

X_val = torch.from_numpy(raw_X_val).float().to(DEVICE)
y_val = torch.from_numpy(raw_y_val).reshape(500, 1).float().to(DEVICE)

# model hyperparameters

alpha = 0.001
nstep = 2500

# fit the polynomial to training dataset

fitted_polynomial, loss_list = fit_polynomial(trn_data=(X_trn, y_trn), 
                                              val_data=(X_val, y_val), hyp_parm=(alpha, nstep))
print(f"resulting weights: {fitted_polynomial.weight}")                                              

print("""learning completed
================================================================================================================""")


#================================================================================================================
# task 7: Find a suitable learning rate and number of iterations for gradient descent. 
#================================================================================================================
print("""starting task 7...
benchmarking models with different hyperparameters. 
please uncomment lines 359, 369, 380, and 393 
as it requires 6 min. each process...""")

# defining the hyperparameters candidates to try

alpha_i = lambda i : (0.1) ** (i/2) if i % 2 == 0 else 0.5 * (0.1) ** ((i - 1) / 2)    
# i: 0 -> 20 => alpha_i: 1 -> 0.00016

nstep_i = lambda i : 250 * i 
# i: 0 -> 20 => alpha_i: 0 -> 5000  

# benchmark the training model with different hyperparameters

print("(if line 359 is uncommented) plotting benchmark heatmap 1/4.")
# benchmarks = benchmark_hyp_parm(alpha_fn=alpha_i, nstep_fn=nstep_i, num_tries=(21, 21), plot_loss_up_to=10, ylabs=["1", "0.5", "0.1", "0.05", "0.01", "0.005", "0.001", "0.0005", "0.0001", "0.00005", "0.00001", "0.000005", "0.000001", "0.0000005", "0.0000001", "0.00000005", "0.00000001", "0.000000005","0.000000001","0.0000000005", "0.0000000001",])

# fine-tunning the new alpha candidates

alpha_i = lambda i : 0.0005 + 0.00005 * i
# i: 0 -> 10 => alpha_i: 0.0005 -> 0.0015 

# benchmark the training model with different hyperparameters, second try

print("(if line 369 is uncommented) plotting benchmark heatmap 2/4")
# benchmarks = benchmark_hyp_parm(alpha_fn=alpha_i, nstep_fn=nstep_i, num_tries=(21, 21), plot_loss_up_to=5,ylabs=["0.00050","0.00055","0.00060","0.00065","0.00070","0.00075","0.00080","0.00085","0.00090","0.00095","0.00100","0.00105","0.00110","0.00115","0.00120","0.00125","0.00130","0.00135","0.00140","0.00145","0.00150"])
# ~ 6 min

# fine-tunning the new alpha candidates

alpha_i = lambda i : 0.0012 + 0.000005 * i 
# i: 0 -> 20 => alpha_i: 0.0012 -> 0.0013

# benchmark the training model with different hyperparameters, third try

print("(if line 380 is uncommented) plotting benchmark heatmap 3/4")
# benchmarks = benchmark_hyp_parm(alpha_fn=alpha_i, nstep_fn=nstep_i, num_tries=(21, 21), plot_loss_up_to=1,ylabs=["0.001200", "0.001205", "0.001210","0.001215","0.001220","0.001225","0.001230", "0.001235", "0.001240","0.001245","0.001250","0.001255","0.001260", "0.001265", "0.001270","0.001275","0.001280","0.001285","0.001290","0.001295","0.001300"])
# ~ 6 min

# fine-tunning the new nstep candidates

alpha_i = lambda i : 0.001254 + i - i #add and sum x for broadcasting in function

nstep_i = lambda i : 100 * i 
# i: 0 -> 20 => alpha_i: 0 -> 1000 

# benchmark the training model with different hyperparameters, fourth try

print("(if line 393 is uncommented) plotting benchmark heatmap 4/4")
# benchmarks = benchmark_hyp_parm(alpha_fn=alpha_i, nstep_fn=nstep_i, num_tries=(1, 21), plot_loss_up_to=0.6, ylabs=["0.001254"])

print("""benchmark completed
================================================================================================================""")


#================================================================================================================
# task 8. Plot the training and validation losses wrt steps
#================================================================================================================
print("""starting task 8...
learning weights through training dataset using optimal hyperparameters""")

# optimal model hyperparameters (losses ~ 0.6)

alpha = 0.001254
nstep = 1600

# fit the polynomial to training dataset
fitted_polynomial, (trn_loss_list, val_loss_list) = fit_polynomial(trn_data=(X_trn, y_trn), 
                                                                   val_data=(X_val, y_val), 
                                                                   hyp_parm=(alpha, nstep))

print("""learning completed
================================================================================================================""")


#================================================================================================================
# task 9. Visualization of the polynomial defined by the estimate of w
#================================================================================================================
print("""starting task 9...
plotting the learned polynomial""")

x = np.linspace(-3, 3, 100)
est_par = torch.squeeze(fitted_polynomial.weight).detach().numpy()

p_x = np.polynomial.polynomial.polyval(x, parameter)
p_x_est = np.polynomial.polynomial.polyval(x, est_par)


fig = plt.figure()
fig.set_size_inches(5, 4)

plt.plot(x, p_x, color='crimson', label="Generator polynomial", alpha=0.6)
plt.plot(x, p_x_est, color='steelblue', label="Estimated polynomial", alpha=0.6)
        
plt.title('Estimated and Genertor Polynomials Compared')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('fig8-model-vs-truth.png', dpi=300)
#plt.show()  

print("""plot done
================================================================================================================""")


#================================================================================================================
# TASK 10. Report and explain what happens when only 10 point in training
#================================================================================================================
print("""starting task 10...
training model with small datasets""")

indexes = np.random.choice(500, size=10)

raw_sm_X_trn = raw_X_trn[indexes, :]
raw_sm_y_trn = raw_y_trn[indexes]

sm_X_trn = torch.from_numpy(raw_sm_X_trn).float().to(DEVICE)
sm_y_trn = torch.from_numpy(raw_sm_y_trn).reshape(10, 1).float().to(DEVICE)

# former optimal model hyperparameters (losses ~ 0.6)

print(sm_X_trn.dtype)
print(X_val.dtype)

alpha = 0.001

# fit the polynomial to training dataset
fit_poly_sm_set_1000its, _ = fit_polynomial(trn_data=(sm_X_trn, sm_y_trn), val_data=(X_val, y_val), hyp_parm=(alpha, 1000))
fit_poly_sm_set_5000its, _ = fit_polynomial(trn_data=(sm_X_trn, sm_y_trn), val_data=(X_val, y_val), hyp_parm=(alpha, 5000))
fit_poly_sm_set_10000its, _ = fit_polynomial(trn_data=(sm_X_trn, sm_y_trn), val_data=(X_val, y_val), hyp_parm=(alpha, 10000))

print(fit_poly_sm_set_1000its.weight)

est_par_sm_set_1000its = torch.squeeze(fit_poly_sm_set_1000its.weight).detach().numpy()
est_par_sm_set_5000its = torch.squeeze(fit_poly_sm_set_5000its.weight).detach().numpy()
est_par_sm_set_10000its = torch.squeeze(fit_poly_sm_set_10000its.weight).detach().numpy()

z = np.linspace(-3, 3, 100)

print("plotting results")

fig = plt.figure()
fig.set_size_inches(5, 4)

plt.scatter(raw_sm_X_trn[:,1], raw_sm_y_trn, color='steelblue', marker='o', s=20, label='Training dataset element')
plt.plot(z, np.polynomial.polynomial.polyval(z, parameter), "steelblue", label='Generator Polynomial')
plt.plot(z, np.polynomial.polynomial.polyval(z, est_par_sm_set_1000its), "crimson", label='Estimated Polynomial after 1000 iterations')
plt.plot(z, np.polynomial.polynomial.polyval(z, est_par_sm_set_5000its), "orange", label='Estimated Polynomial after 5000 iterations')
plt.plot(z, np.polynomial.polynomial.polyval(z, est_par_sm_set_10000its), "violet", label='Estimated Polynomial after 10000 iterations')

plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend(loc="upper center")
plt.savefig('fig9-learning-small-data.png', dpi=300)

print("""plot done
================================================================================================================""")


#================================================================================================================
# task 11. Plot the evolution of each coefficient of w as a function of the gradient descent iterations.
#================================================================================================================
print("""starting task 11...
studying the evolution of the weights""")

# model hyperparameters

alpha = 0.001254
nstep = 3000

par_evolution = fit_polynomial(trn_data=(X_trn, y_trn), 
                               val_data=(X_val, y_val), 
                               hyp_parm=(alpha, nstep), 
                               study_weights=True)

plot_weight_evolution(par_evolution)

print(f"""plots done.
model weights: {fitted_polynomial.weight}
================================================================================================================
Assignment 1 finished""")
