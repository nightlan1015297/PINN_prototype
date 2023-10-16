

class Pinns:
    def __init__(self, n_int_):
        self.n_int = n_int_


        # Initial condition to solve driven oscillator
        self.initial_x = 5
        self.initial_v = 0
        self.init_cond = torch.tensor([self.initial_x,self.initial_v])
        
        
        self.omega_o = 5
        self.omega_d = 15
        self.gamma   = 0.3
        self.mass = 1
        self.F_o = 5
        
        self.lambda_u = 50
        
        # Extrema of the solution domain (t) in [0,5]
        self.domain_extrema = torch.tensor([[0, 100]])  # Space dimension


        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                            n_hidden_layers=4,
                                            neurons=20,
                                            regularization_param=0.,
                                            regularization_exp=2.,
                                            retrain_seed=42)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=1)

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[1] - self.domain_extrema[0]) + self.domain_extrema[0]

    def driving_term(self, t):
        return self.F_o*torch.sin(*self.omega_d*t)/self.mass
    
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_int, output_int = self.add_interior_points()         # S_int
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_int
    

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self):
        x_init = torch.zeros(1)
        x_init.reauires_grad = True
        pred_x_init = self.approximate_solution(x_init)
        grad_u = torch.autograd.grad(pred_x_init.sum(), x_init, create_graph=True)
        pred_v_init = grad_u[0]
        return torch.tensor([pred_x_init, pred_v_init])

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

        residual = grad_u_tt + self.gamma*grad_u_t + self.omega_o**2*u - self.driving_term(input_int)
        return residual.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_int, verbose=True):
        u_pred_tb = self.apply_initial_condition()

        assert (u_pred_tb.shape[1] == self.init_cond.shape[1])


        r_int = self.compute_pde_residual(inp_train_int)
        r_tb = self.init_cond - u_pred_tb

        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)

        loss = torch.log10(self.lambda_u *  loss_tb + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_tb).item(), 4), "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")
            
            for j, (inp_train_int, u_train_int) in enumerate(self.training_set_int):
                def closure():
                    optimizer.zero_grad()
                    
                    loss = self.compute_loss(inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    ################################################################################################
    def plotting(self):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=exact_output.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Exact Solution")
        axs[1].set_title("Approximate Solution")

        plt.show()

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")
