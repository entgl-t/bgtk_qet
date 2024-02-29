import pennylane as qml
import torch
import math
from bgtk_qet_sp.utils import QSP_circuit

torch_pi_4 = torch.Tensor([math.pi/4])


class QSP_Circ_Fit(torch.nn.Module):
    def __init__(self, degree, num_vals):
        super().__init__()
        #self.phi = torch_pi_4 * torch.rand(degree + 1, requires_grad=True, generator=gen)
        self.phi = torch_pi_4 * torch.tensor([1]+[0 for d in range(degree-1)] + [1], requires_grad=True,dtype=torch.float)
        self.phi = torch.nn.Parameter(self.phi)
        self.num_phi = degree + 1
        self.num_vals = num_vals


    def forward(self, omega_mats):
        y_pred = []
        generate_qsp_mat = qml.matrix(QSP_circuit)
        for w in omega_mats:
            u_qsp = generate_qsp_mat(self.phi, w)
            P_a = u_qsp[0, 0]  # the (0,0) element of the matrix corresponds to measur. <0|U|0>
            y_pred.append(P_a.real)
        return torch.stack(y_pred, 0)