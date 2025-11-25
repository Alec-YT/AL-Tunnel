import torch
import torch.nn as nn

# Individual neural networks 
class NN(nn.Module):
    def __init__(self, layers, dropout_rate=0.05):
        super(NN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        n_hidden = len(layers) - 2  

        for i in range(len(layers) - 1):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))

            # First two layers 
            if 0 <= i < 2 and i < n_hidden:  
                self.dropouts.append(nn.Dropout(p=dropout_rate))
            else:
                self.dropouts.append(None)  

        self.output_layer = nn.Linear(layers[-1], 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = torch.tanh(layer(x))
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x

class PINN_model(nn.Module):
    def __init__(self, layers, nhv=None, nh=None, beta=None):
        super(PINN_model, self).__init__()
        
        # --- NN parameters ---
        self.ux_params = NN(layers)
        self.uy_params = NN(layers)

        # --- Initialisation aléatoire dans l'intervalle ---
        self.Eh_normalized = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.Ev_normalized = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.Gvh_normalized = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.K = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.beta_ = nn.Parameter(torch.rand(1, dtype=torch.float32))

        self.register_buffer("nhv", torch.tensor([nhv], dtype=torch.float32))
        self.register_buffer("nh", torch.tensor([nh], dtype=torch.float32))
        # self.register_buffer("beta_", torch.tensor([beta], dtype=torch.float32))

    def forward(self, x):
        ux = self.ux_params(x)
        uy = self.uy_params(x)
        return ux, uy
        


