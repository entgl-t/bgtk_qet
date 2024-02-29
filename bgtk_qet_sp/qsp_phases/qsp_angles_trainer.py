import matplotlib.pyplot as plt
import torch

class QSP_Model_Trainer:
    def __init__(self, model, degree, num_samples, a_vals, process_theta_vals, y_true,threshold=0.5,f_type='gauss', optim_params=('sgd',1e-5, 0.5, 0.999,'sum')):
        self.model = model
        self.degree = degree
        self.num_samples = num_samples
        self.a_vals = a_vals
        self.inp = process_theta_vals(self.a_vals)
        self.y_true = y_true
        self.threshold = threshold
        self.f_type = f_type
        self.optim_params = optim_params
        #self.random_seed = 63647585

    def execute_bfgs(self,  num_iter=25000, lr = 1e-3, history_size= 10,max_iter = 4,verbose=True):
        model = self.model(degree=self.degree, num_vals=self.num_samples)
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=history_size, max_iter=max_iter, lr=lr)
        t = 0
        loss_val = 1.0
        input = self.inp
        y_true = self.y_true
        while (t <= num_iter) and (loss_val > self.threshold):
            #y_pred = model(self.inp)

            if t == 1:
                self.init_y_pred =  model(self.inp)

            def closure():
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                y_pred = model(input)

                # Compute loss
                loss = criterion(y_pred, y_true)

                # Backward pass
                loss.backward()

                return loss
            # Update weights
            optimizer.step(closure)

            # Update the running loss
            loss = closure()
            loss_val = loss.item()
            if (t % 200== 0) and verbose:
                print(f"---- iter: {t}, loss: {round(loss_val, 10)} -----")
            if (t % 3000== 0):
                self.save_angles(model.phi, loss_val)

            t += 1
        self.save_angles(model.phi, loss_val)

    def execute(self,  num_iter=25000,verbose=True ):
        model = self.model(degree=self.degree, num_vals=self.num_samples)

        criterion = torch.nn.MSELoss(reduction=self.optim_params[4])
        #optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,momentum=0.9)
        if self.optim_params[0] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.optim_params[1], momentum=0.9, nesterov=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.optim_params[1], betas=(self.optim_params[2], self.optim_params[3]))

        t = 0
        loss_val = 1.0
        while (t <= num_iter) and (loss_val > self.threshold):
            self.y_pred = model(self.inp)

            if t == 1:
                self.init_y_pred = self.y_pred

            # Compute and print loss
            loss = criterion(self.y_pred, self.y_true)
            loss_val = loss.item()

            if(t % 200 == 0) and verbose:
                print(f"---- iter: {t}, loss: {round(loss_val, 10)} -----")
            if (t % 1000 == 0):
                self.save_angles(model.phi, loss_val)

            # Perform a backward pass and update weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t += 1
        self.save_angles(model.phi ,loss_val )


    def save_angles(self,model_params,threshold):
        torch.save(model_params, self.f_type +'_qsp_angles_deg_' + str(self.degree) + '_error_'+str(threshold)+'_num_sampl_'+str(self.num_samples)+'.pt')

    def plot_result(self, show=True):
        """Plot the results"""
        plt.plot(self.a_vals, self.y_true.tolist(), "--b", label="target func")
        plt.plot(self.a_vals, self.y_pred.tolist(), ".g", label="optim params")
        plt.plot(self.a_vals, self.init_y_pred.tolist(), ".r", label="init params")
        plt.legend(loc=1)

        if show:
            plt.show()

