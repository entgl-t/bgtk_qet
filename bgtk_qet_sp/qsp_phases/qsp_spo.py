import pennylane as qml
import torch
import math
from bgtk_qet_sp.qet_state_prep.utils import QSP_circ

torch_pi_4 = torch.Tensor([math.pi/4])


class QSP_Func_Fit(torch.nn.Module):
    def __init__(self, degree, num_vals):
        """Given the degree and number of samples, this method randomly
        initializes the parameter vector (randomness can be set by random_seed)
        """
        super().__init__()

        #self.phi = torch_pi_4 * torch.rand(degree + 1, requires_grad=True, generator=gen)
        self.phi = torch_pi_4 * torch.tensor([1]+[0 for d in range(degree-1)] + [1], requires_grad=True,dtype=torch.float)

        self.phi = torch.nn.Parameter(self.phi)
        self.num_phi = degree + 1
        self.num_vals = num_vals




    def forward(self, omega_mats):
        """PennyLane forward implementation"""
        y_pred = []
        generate_qsp_mat = qml.matrix(QSP_circ)


        for w in omega_mats:
            u_qsp = generate_qsp_mat(self.phi, w)
            P_a = u_qsp[0, 0]  # Taking the (0,0) entry of the matrix corresponds to <0|U|0>
            y_pred.append(P_a.real)


        return torch.stack(y_pred, 0)