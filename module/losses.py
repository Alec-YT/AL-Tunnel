import torch
import torch.nn as nn

def computes_physics_residuals(model, ux_int=None, uy_int=None, inputs_int=None, 
                               ux_bc=None, uy_bc=None, inputs_bc=None, device=torch.device("cpu")):

    # Default
    eq1 = torch.tensor([], device=device)
    eq2 = torch.tensor([], device=device)
    eq3 = torch.tensor([], device=device)
    eq4 = torch.tensor([], device=device)

    # =============================
    # Equilibrium equations
    # =============================
    if ux_int is not None and inputs_int is not None and len(inputs_int) > 0:
        # 1er ordre
        grads_ux = torch.autograd.grad(ux_int, inputs_int, grad_outputs=torch.ones_like(ux_int),
                                       create_graph=True)[0]
        ux_x = grads_ux[:, 0]
        ux_y = grads_ux[:, 1]

        grads_uy = torch.autograd.grad(uy_int, inputs_int, grad_outputs=torch.ones_like(uy_int),
                                       create_graph=True)[0]
        uy_x = grads_uy[:, 0]
        uy_y = grads_uy[:, 1]

        # 2e order
        grads_ux_x = torch.autograd.grad(ux_x, inputs_int, grad_outputs=torch.ones_like(ux_x),
                                         create_graph=True)[0]
        grads_ux_y = torch.autograd.grad(ux_y, inputs_int, grad_outputs=torch.ones_like(ux_y),
                                         create_graph=True)[0]
        grads_uy_x = torch.autograd.grad(uy_x, inputs_int, grad_outputs=torch.ones_like(uy_x),
                                         create_graph=True)[0]
        grads_uy_y = torch.autograd.grad(uy_y, inputs_int, grad_outputs=torch.ones_like(uy_y),
                                         create_graph=True)[0]

        ux_xx = grads_ux_x[:, 0]
        ux_xy = grads_ux_x[:, 1]
        ux_yy = grads_ux_y[:, 1]
        uy_xx = grads_uy_x[:, 0]
        uy_xy = grads_uy_y[:, 0]
        uy_yy = grads_uy_y[:, 1]

        # Coefficients
        D = - model.Eh_normalized / (2 * model.nhv**2 * (1 + model.nh) - model.Eh_normalized/model.Ev_normalized * (1 - model.nh**2))

        # Residual
        eq1 = D * (model.nhv*(1+model.nh) * uy_xy + (model.Eh_normalized/model.Ev_normalized - model.nhv**2) * ux_xx) \
              + model.Gvh_normalized * (ux_yy + uy_xy)

        eq2 = D * (model.nhv*(1+model.nh) * ux_xy + (1 - model.nh**2) * uy_yy) \
              + model.Gvh_normalized * (ux_xy + uy_xx)


    # =============================
    # Boundary conditions
    # =============================
    if ux_bc is not None and inputs_bc is not None and len(inputs_bc) > 0:
        x_b = inputs_bc[:, 0]
        y_b = inputs_bc[:, 1]

        grads_ux_bc = torch.autograd.grad(ux_bc, inputs_bc, grad_outputs=torch.ones_like(ux_bc),
                                          create_graph=True, allow_unused=True)[0]
        grads_uy_bc = torch.autograd.grad(uy_bc, inputs_bc, grad_outputs=torch.ones_like(uy_bc),
                                          create_graph=True, allow_unused=True)[0]

        ux_x_b = grads_ux_bc[:, 0]
        ux_y_b = grads_ux_bc[:, 1]
        uy_x_b = grads_uy_bc[:, 0]
        uy_y_b = grads_uy_bc[:, 1]

        D = - model.Eh_normalized / (2 * model.nhv**2 * (1 + model.nh) - model.Eh_normalized/model.Ev_normalized * (1 - model.nh**2))

        eq3 = x_b * D * (model.nhv*(1+model.nh)*uy_y_b + (model.Eh_normalized/model.Ev_normalized - model.nhv**2)*ux_x_b) \
              + x_b * 0.5 * (1 + model.K - (1 - model.K) * torch.cos(2*model.beta_)) \
              + y_b * model.Gvh_normalized * (ux_y_b + uy_x_b) \
              + y_b * 0.5 * (model.K - 1) * torch.sin(2*model.beta_) 

        eq4 = y_b * D * (model.nhv*(1+model.nh)*ux_x_b + (1 - model.nh**2)*uy_y_b) \
              + y_b * 0.5 * (1 + model.K + (1 - model.K) * torch.cos(2*model.beta_)) \
              + x_b * model.Gvh_normalized * (ux_y_b + uy_x_b) \
              + x_b * 0.5 * (model.K - 1) * torch.sin(2*model.beta_) 


    return eq1, eq2, eq3, eq4



def pinn_loss(model, batch, data=True, residuals=True, test=False, device=torch.device("cpu")):

    # Unpacking
    inputs = batch['inputs']
    true_values = batch['outputs']
    is_int = batch['is_int']
    is_bc = batch['is_bc']
    is_data = batch['is_data']

    # Initialize loss scalaires
    total_loss = torch.tensor(0.0, device=device)
    loss_eq1 = torch.tensor(0.0, device=device)
    loss_eq2 = torch.tensor(0.0, device=device)
    loss_eq3 = torch.tensor(0.0, device=device)
    loss_eq4 = torch.tensor(0.0, device=device)
    loss_ux  = torch.tensor(0.0, device=device)
    loss_uy  = torch.tensor(0.0, device=device)

    mse_loss = nn.MSELoss()
    
    # Weights
    if test:
        w_pde = w_bc = w_data = 1.0
    else:
        w_pde, w_bc, w_data = 1.0, 10.0, 100.0

    # ======================
    # PDE residuals 
    # ======================
    if residuals:
        model.eval()  
        
        # Prepare inputs for equilibrium equations
        tensors_to_cat = [] 
        if is_int.any():
            tensors_to_cat.append(inputs[is_int])
        if is_bc.any():
            tensors_to_cat.append(inputs[is_bc])
            
        if tensors_to_cat:
            inputs_int = torch.cat(tensors_to_cat, dim=0).requires_grad_(True)  
            ux_int, uy_int = model(inputs_int)  
            ux_int, uy_int = ux_int.squeeze(-1), uy_int.squeeze(-1)
        else:
            ux_int = uy_int = inputs_int = None
            
        # Boundary conditions
        if is_bc.any():
            inputs_bc = inputs[is_bc].requires_grad_(True)
            ux_bc, uy_bc = model(inputs_bc)  # Forward pass WITHOUT dropout
            ux_bc, uy_bc = ux_bc.squeeze(-1), uy_bc.squeeze(-1)
        else:
            ux_bc = uy_bc = inputs_bc = None

        # Compute residuals
        eq1_residuals, eq2_residuals, eq3_residuals, eq4_residuals = computes_physics_residuals(
            model, ux_int, uy_int, inputs_int, ux_bc, uy_bc, inputs_bc, device=device
        )

        # Compute losses
        if eq1_residuals is not None and eq1_residuals.numel() > 0:
            loss_eq1 = w_pde * mse_loss(eq1_residuals, torch.zeros_like(eq1_residuals))
            total_loss = total_loss + loss_eq1
        if eq2_residuals is not None and eq2_residuals.numel() > 0:
            loss_eq2 = w_pde * mse_loss(eq2_residuals, torch.zeros_like(eq2_residuals))
            total_loss = total_loss + loss_eq2
        if eq3_residuals is not None and eq3_residuals.numel() > 0:
            loss_eq3 = w_bc * mse_loss(eq3_residuals, torch.zeros_like(eq3_residuals))
            total_loss = total_loss + loss_eq3
        if eq4_residuals is not None and eq4_residuals.numel() > 0:
            loss_eq4 = w_bc * mse_loss(eq4_residuals, torch.zeros_like(eq4_residuals))
            total_loss = total_loss + loss_eq4

    # ======================
    # Data fitting
    # ======================
    if data and is_data.any():
        model.train()  
        
        inputs_data = inputs[is_data].requires_grad_(False)
        ux_data, uy_data = model(inputs_data)  # Forward pass WITH dropout
        ux_data, uy_data = ux_data.squeeze(-1), uy_data.squeeze(-1)
        
        ux_true, uy_true = true_values[is_data, 0], true_values[is_data, 1]
        loss_ux = w_data * mse_loss(ux_data, ux_true)
        loss_uy = w_data * mse_loss(uy_data, uy_true)
        total_loss = total_loss + loss_ux + loss_uy

    return (total_loss,
            loss_eq1, loss_eq2, loss_eq3, loss_eq4,
            loss_ux, loss_uy)
