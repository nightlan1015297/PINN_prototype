#!/usr/bin/env python
# coding: utf-8

#%%
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go

torch.manual_seed(5)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#%%
class SinActivation(nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        
    def forward(self, x):
        return torch.sin(x)
    
class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = SinActivation()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss



class DrivenOscillatorPinns:
    def __init__(self, n_int_):
        self.n_int = n_int_


        # Initial condition to solve driven oscillator
        self.initial_x = 0
        self.initial_v = 0
        self.init_cond = torch.tensor([self.initial_x,self.initial_v])
        
        self.k = 50
        self.mass = .5

        self.omega_d = 9.95
        self.phid = 1 
        
        self.c = 0.01
        self.F_o = 50
        
        self.initial_weight  = 1
        self.residual_weight = 5
        self.trivial_killer_weight = 1
        
        # Extrema of the solution domain (t) in [0,5]
        self.domain_extrema = torch.tensor([[0, 25]])  
        
        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=1)
        

        self.training_set_int = self.assemble_datasets()

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                            n_hidden_layers=3,
                                            neurons=32,
                                            regularization_param=0.,
                                            regularization_exp=0.,
                                            retrain_seed=5).to(device)
        
        
    def driven_oscillator(self,t):
        w0 = np.sqrt(self.k/self.mass)
        gamma = self.c/2/self.mass
        
        wprime = np.sqrt(w0**2 - gamma**2)
        print(f"wprime = {wprime}")
        A = self.F_o / self.mass / np.sqrt((w0**2 - self.omega_d**2)**2 + 4 * gamma**2 * self.omega_d**2 )
        
        print(self.k, self.mass, self.omega_d**2)
        phi = np.arctan(self.c * self.omega_d / (self.k - self.mass * self.omega_d**2)) - self.phid
        phih = np.arctan(wprime * (self.initial_x - A * np.cos(phi)) / (self.initial_v + gamma * (self.initial_x - A * np.cos(phi)) - A * self.omega_d * np.sin(phi) ) )
        
        Ah = (self.initial_x - A * np.cos(phi)) / np.sin(phih)
        
        x = Ah * np.exp(-gamma * t) * np.sin(wprime * t + phih) + A * np.cos(self.omega_d * t - phi)
        
        return x

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[0][1] - self.domain_extrema[0][0]) + self.domain_extrema[0][0]

    def driving_term(self, t):
        return self.F_o*torch.cos(self.omega_d*t + self.phid)
    
    def exact_solution(self, t):
        return
    
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int.to(device), output_int.to(device)

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_int, output_int = self.add_interior_points()         # S_int
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_int


    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def initial_velocity(self,init_t):
        init_t.requires_grad = True
        pred_x_init = self.approximate_solution(init_t)
        pred_v_init = torch.autograd.grad(pred_x_init.sum(), init_t, create_graph=True)[0]
        
        return pred_v_init
    
    def initial_position(self,init_t):

        pred_x_init = self.approximate_solution(init_t)
        
        return pred_x_init

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_u_t = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_tt = torch.autograd.grad(grad_u_t.sum(), input_int, create_graph=True)[0]

        #grad_u_sq_x = torch.autograd.grad(u_sq.sum(), input_int, create_graph=True)[0][:,1]

        residual = self.mass * grad_u_tt  + self.k*u # - self.driving_term(input_int) + self.gamma*grad_u_t
        return residual.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_int ,verbose=True):
        
        init_t = torch.zeros(1).to(device)
        pred_init_position = self.initial_position(init_t)
        pred_init_velocity = self.initial_velocity(init_t)
        

        # assert (u_pred_tb.shape[0] == self.init_cond.shape[0])

        inp_train_int.requires_grad = True
        u = self.approximate_solution(inp_train_int)

        grad_u_t = torch.autograd.grad(u.sum(), inp_train_int, create_graph=True)[0]
        grad_u_tt = torch.autograd.grad(grad_u_t.sum(), inp_train_int, create_graph=True)[0]


        residual = self.mass*grad_u_tt + self.c*grad_u_t + self.k*u  - self.driving_term(inp_train_int) 
        
        
        init_position_error = self.init_cond[0] - pred_init_position
        init_velocity_error = self.init_cond[1] - pred_init_velocity
        
        loss_initial_position = torch.mean(init_position_error ** 2)
        loss_initial_velocity = torch.mean(init_velocity_error ** 2)

        loss_int = torch.mean(residual ** 2) 
        loss_tb = loss_initial_position + loss_initial_velocity
        loss_trivial = 0
        # loss_trivial = (torch.reciprocal(torch.mean(u**2)) + torch.reciprocal(torch.mean(grad_u_t**2)) + torch.reciprocal(torch.mean(grad_u_tt**2)))*self.trivial_killer_weight
        loss = torch.log10(loss_tb * self.initial_weight  + loss_int * self.residual_weight) + loss_trivial * self.trivial_killer_weight
        
        if verbose: print("Total loss: ", round(loss.item(), 8), 
                        "| PDE Loss: ", round(loss_int.item(), 8), 
                        "| Initial Loss: ", round(loss_tb.item(), 8),
                        "| Trivial Loss: ", round(loss_trivial.item(), 8))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        #  prograssive training
        # for fuck in np.linspace(0,25,5):
        #     self.domain_extrema = torch.tensor([[0, fuck]])  
        #     self.training_set_int = self.assemble_datasets()
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")
            
            for _, (inp_train_int, u_train_int) in enumerate(self.training_set_int):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_int ,verbose=verbose)
                    loss.backward()
                    
                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Mnimum Loss: ', min(history))

        return history


#%%
n_int = 1000
pinn = DrivenOscillatorPinns(n_int)

input_int_, output_int_ = pinn.add_interior_points()

# optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
#                             lr=float(0.001),
#                             max_iter=50000,
#                             max_eval=50000,
#                             history_size=150,
#                             line_search_fn="strong_wolfe",
#                               tolerance_change=1.0 * np.finfo(float).eps)


#%%
hist = []


optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                            lr=float(0.001))

pinn.initial_weight        = 1.0
pinn.residual_weight       = 1.0

hist += pinn.fit(num_epochs=10000,
                optimizer=optimizer_ADAM,
                verbose=True)


#%%
fig = go.Figure()
fig.add_trace(go.Scatter(y=hist))
# set dark theme
fig.update_layout(title='Loss function', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark')
fig.show()


#%%
inputs = pinn.soboleng.draw(1000)
inputs = pinn.convert(inputs)
test = pinn.approximate_solution.to('cpu')
output = test(inputs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=inputs.detach().numpy()[:,0], y=output.detach().numpy()[:,0], mode='markers', name='PINN solution', marker=dict(color='yellow')))
#set marker size
fig.add_trace(go.Scatter(x=inputs.detach().numpy()[:,0], y=pinn.driven_oscillator(inputs).detach().numpy()[:,0], mode='markers', name='Exact solution', marker=dict(color='blue',size=3)))
fig.update_layout(title='Exact solution vs PINN solution', xaxis_title='time', yaxis_title='position', template='plotly_dark')

#fig.write_image('oscillator_pinn.png')
fig.show()


#%%
# print initial velocity and position
init_t = torch.zeros(1)
pred_init_position = pinn.initial_position(init_t)
pred_init_velocity = pinn.initial_velocity(init_t)
print("Initial position: ", pred_init_position.item())
print("Initial velocity: ", pred_init_velocity.item())

# %%
