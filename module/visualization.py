import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_dataset_points(dataset, R, x_max, save_path=None):

    # Collect inputs and masks
    inputs = dataset.inputs.cpu().detach().numpy()
    x_all, y_all = inputs[:, 0], inputs[:, 1]

    is_int = dataset.is_int.cpu().numpy()
    is_bc = dataset.is_bc.cpu().numpy()
    is_data = dataset.is_data.cpu().numpy()

    s = 30
    plt.figure(figsize=(8, 8))

    # Tunnel plot
    theta = torch.linspace(0, 2*torch.pi, 100)
    x_circle = (R * torch.cos(theta) / x_max).cpu().numpy()
    y_circle = (R * torch.sin(theta) / x_max).cpu().numpy()
    plt.plot(x_circle, y_circle, 'r-', linewidth=1, label="Tunnel wall", zorder=5)

    # BC points
    plt.scatter(x_all[is_bc], y_all[is_bc], s=2*s, color='darkgreen', label='Boundary', alpha=0.6, zorder=3)

    # Interior points
    plt.scatter(x_all[is_int], y_all[is_int], s=2*s, color='blue', label='Interior', alpha=0.6, zorder=2)

    # Measurement points
    plt.scatter(x_all[is_data], y_all[is_data], s=s, color='black', marker='x', label='Measurements', linewidths=1, zorder=4)

    plt.legend(loc='upper right')
    plt.xlabel(r"$\tilde x$")
    plt.ylabel(r"$\tilde y$")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'bc_int_data_points' + '.pdf'), bbox_inches='tight', dpi=300)
    plt.show()



def relative_error(opt,true):
    return np.round(np.abs(true-opt)/np.abs(true)*100, 1)



def plot_predictions_vs_true(model, test_dataset, save_dir=None, n_samples=50, device=torch.device("cpu")):
    """
    Plot true vs predicted displacements using Monte Carlo Dropout.
    Performs n_samples stochastic forward passes ,
    averages predictions, and computes epistemic uncertainty.
    """
    model.train()  
    inputs = test_dataset.inputs.to(device)
    true_values = test_dataset.outputs.to(device)

    preds_all = []

    with torch.no_grad():
        for _ in range(n_samples):
            preds = model(inputs)
            if isinstance(preds, tuple):
                preds = torch.cat([p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in preds], dim=1)
            preds_all.append(preds.unsqueeze(0))

    preds_all = torch.cat(preds_all, dim=0)  
    preds_mean = preds_all.mean(dim=0)

    preds_np = preds_mean.cpu().numpy()
    true_np = true_values.cpu().numpy()

    if np.isnan(preds_np).any():
        num_outputs = preds_np.shape[1]
        return [np.nan] * num_outputs

    assert preds_np.shape == true_np.shape, f"Shape mismatch: preds {preds_np.shape}, true_values {true_np.shape}"

    num_cols = preds_np.shape[1]
    names = [r'$u_{x}$', r'$u_{y}$']  
    all_r2 = []

    # Save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(num_cols):
        col_preds = preds_np[:, i]
        col_true = true_np[:, i]

        # Linear Regression Coef
        r2 = round(r2_score(col_true, col_preds), 4)
        all_r2.append(r2)

        # Linear regression
        lr = LinearRegression()
        lr.fit(col_true.reshape(-1, 1), col_preds)

        # Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(col_true, col_preds, alpha=0.7)
        plt.plot(col_true, lr.predict(col_true.reshape(-1, 1)), color='red', linestyle="-", label=fr"$R^2 = {r2}$")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(names[i] if i < len(names) else f"Output {i+1}")
        plt.legend()
        plt.grid(True)

        # Save
        if save_dir:
            varname = names[i].replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
            filename = f"r2_MC_{varname}.pdf"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

    return all_r2


def plot_training_diagnostics(train_total_loss, train_loss_eq1, train_loss_eq2,
                               train_loss_eq3, train_loss_eq4, train_loss_ux, train_loss_uy, scale,
                               nadam_epochs=0, 
                               Eh_opt=None, Ev_opt=None, Gvh_opt=None, K_opt=None, beta_opt=None,
                               Eh_true=None, Ev_true=None, Gvh_true=None, K_true=None, beta_true=None,
                               show_error=True, plot_losses=True, plot_params=True, save_path=None):
    
    epochs = nadam_epochs 
    starting_epoch = 0
    ending_epoch = epochs
    loss_0 = train_total_loss[0]

    def save_plot(filename):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')

    if plot_losses:
        plt.figure(figsize=(7,5))
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_total_loss], label='Total train')
        if nadam_epochs !=0:
            plt.axvspan(starting_epoch, nadam_epochs, color='blue', alpha=0.05, label='Nadam region')
            plt.axvspan(nadam_epochs, ending_epoch, color='green', alpha=0.05, label='L-BFGS region')
        plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel('Training losses')
        save_plot('total_loss.pdf'); plt.show()

        plt.figure(figsize=(7,5))
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_loss_eq1], label='Eq 1')
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_loss_eq2], label='Eq 2')
        if nadam_epochs !=0:
            plt.axvspan(starting_epoch, nadam_epochs, color='blue', alpha=0.05, label='Nadam region')
            plt.axvspan(nadam_epochs, ending_epoch, color='green', alpha=0.05, label='L-BFGS region')
        plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel('Training losses')
        save_plot('eq1_eq2_losses.pdf'); plt.show()

        plt.figure(figsize=(7,5))
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_loss_eq3], label='Eq 3')
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_loss_eq4], label='Eq 4')
        if nadam_epochs !=0:
            plt.axvspan(starting_epoch, nadam_epochs, color='blue', alpha=0.05, label='Nadam region')
            plt.axvspan(nadam_epochs, ending_epoch, color='green', alpha=0.05, label='L-BFGS region')
        plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel('Training losses')
        save_plot('eq3_eq4_losses.pdf'); plt.show()

        plt.figure(figsize=(7,5))
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_loss_ux], label=r'$u_x$')
        plt.plot(np.arange(epochs), [loss/loss_0 for loss in train_loss_uy], label=r'$u_y$')
        if nadam_epochs !=0:
            plt.axvspan(starting_epoch, nadam_epochs, color='blue', alpha=0.05, label='Nadam region')
            plt.axvspan(nadam_epochs, ending_epoch, color='green', alpha=0.05, label='L-BFGS region')
        plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel('Training losses')
        save_plot('ux_uy_losses.pdf'); plt.show()

    if plot_params:
        def plot_param(label, opt, true_val, filename, scale=1):
            plt.figure(figsize=(7,5))
            plt.plot(np.arange(epochs), [v/scale for v in opt], color='red', label=f'Optimized {label}')
            plt.axhline(y=true_val, linestyle='--', color='red', label=f'True {label}')
            if nadam_epochs !=0:
                plt.axvspan(starting_epoch, nadam_epochs, color='blue', alpha=0.05, label='Nadam region')
                plt.axvspan(nadam_epochs, ending_epoch, color='green', alpha=0.05, label='L-BFGS region')
            plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel(label)

            # Relative error annotation
            if show_error:
                if opt is not None:
                    scale = scale if label in [r'$E_h$', r'$E_v$', r'$G_{vh}$'] else 1
                    rel_err = relative_error(opt[-1]/scale, true_val)
                    text_str = f'Rel. error: {rel_err:.1f}%'
                    
                    # Positioning: bottom-right corner
                    x_text = 0.95 * epochs
                    y_min, y_max = plt.ylim()
                    y_text = y_min + 0.5 * (y_max - y_min)
                    plt.text(x_text, y_text, text_str, 
                            ha='right', 
                            va='bottom', 
                            color='black', 
                            fontsize=11,
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
                    
            save_plot(filename)
            plt.show()

        for label, opt, true_val, filename in [
            (r'$E_h$', Eh_opt, Eh_true, 'Eh.pdf'),
            (r'$E_v$', Ev_opt, Ev_true, 'Ev.pdf'),
            (r'$G_{vh}$', Gvh_opt, Gvh_true, 'Gvh.pdf'),
            (r'$K$', K_opt, K_true, 'K.pdf'),
            (r'$\beta$', beta_opt, beta_true, 'beta.pdf'),
        ]:
            if opt is not None:
                scale = scale if label in [r'$E_h$', r'$E_v$', r'$G_{vh}$'] else 1
                plot_param(label, opt, true_val, filename, scale)

            if show_error:
                print(f"Relative error {label}: {relative_error(opt[-1]/scale, true_val):.1f}%")





def plot_preds_2d_MC(model, x_max, u_a, R, save_dir=None, prefix=None, n_samples=50, device=torch.device("cpu")):
    """
    Effectue Monte Carlo Dropout pour estimer la moyenne (et écart-type) 
    de ux et uy sur une grille 2D.
    """
    # --- Grille régulière ---
    N = 200
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    xy = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).to(device)

    # --- Échantillonnage MC Dropout ---
    ux_samples = []
    uy_samples = []
    model.train()  
    with torch.no_grad():
        for _ in range(n_samples):
            ux_pred, uy_pred = model(xy)
            ux_pred = -ux_pred.cpu().numpy() * u_a.cpu().numpy()
            uy_pred = -uy_pred.cpu().numpy() * u_a.cpu().numpy()
            ux_samples.append(ux_pred)
            uy_samples.append(uy_pred)

    ux_samples = np.stack(ux_samples, axis=0)  
    uy_samples = np.stack(uy_samples, axis=0)

    # --- Moyenne et écart-type ---
    ux_mean = np.mean(ux_samples, axis=0).reshape(N, N)
    uy_mean = np.mean(uy_samples, axis=0).reshape(N, N)
    ux_std = np.std(ux_samples, axis=0).reshape(N, N)
    uy_std = np.std(uy_samples, axis=0).reshape(N, N)

    # --- Masquage du tunnel ---
    X_scaled, Y_scaled = X * x_max, Y * x_max
    mask = X_scaled**2 + Y_scaled**2 <= R**2
    for arr in [ux_mean, uy_mean, ux_std, uy_std]:
        arr[mask] = np.nan

    # --- Fonction utilitaire pour tracer ---
    def plot_field(Z, title, fname, cmap="RdBu_r"):
        fig, ax = plt.subplots(figsize=(7, 7))
        cf = ax.contourf(X_scaled, Y_scaled, Z, levels=50, cmap=cmap, extend="both")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.3)
        plt.colorbar(cf, cax=cax, label=title)
        circle = plt.Circle((0, 0), R, color="red", fill=False, linewidth=3)
        ax.add_patch(circle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True)
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # --- Tracés ---
    plot_field(ux_mean, r"$\overline{u_x}$", f"ux_mean{('_' + prefix) if prefix else ''}.pdf")
    plot_field(uy_mean, r"$\overline{u_y}$", f"uy_mean{('_' + prefix) if prefix else ''}.pdf")
    plot_field(ux_std, r"$\sigma(u_x)$", f"ux_std{('_' + prefix) if prefix else ''}.pdf", cmap="viridis")
    plot_field(uy_std, r"$\sigma(u_y)$", f"uy_std{('_' + prefix) if prefix else ''}.pdf", cmap="viridis")


def plot_pool_points_article(pool, all_inputs_train_scaled, all_outputs_train_scaled,
                             inputs_grid_scaled, mask_int_selected_grid, mask_bc_selected_grid,
                             sensor_ids_list, R, x_max, save_path=None):

    # ----------------
    #  FIGURE & STYLE
    # ----------------
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7
    })

    fig, axes = plt.subplots(1, 2, figsize=(16/2.54, 8/2.54))  
    s = 12 # point size

    # ---------------------
    # DATASET Measurements
    # ---------------------
    pool_data = add_ext_points_dropout(model=None, pool=pool, sensor_ids_list=sensor_ids_list, device=device)

    dataset_data = PINNDataset(pool_data, include_int=True, include_bc=True, include_data=True, device=device)

    x_all, y_all = dataset_data.inputs.cpu().numpy().T
    is_int = dataset_data.is_int.cpu().numpy()
    is_bc = dataset_data.is_bc.cpu().numpy()
    is_data = dataset_data.is_data.cpu().numpy()

    theta = torch.linspace(0, 2*torch.pi, 100)
    x_circle = (R * torch.cos(theta) / x_max).cpu().numpy()
    y_circle = (R * torch.sin(theta) / x_max).cpu().numpy()

    ax = axes[0]  
    ax.plot(x_circle, y_circle, 'r-', linewidth=1, label="Tunnel wall", zorder=5)
    ax.scatter(x_all[is_bc], y_all[is_bc], s=2*s, color='darkgreen', label="Boundary", alpha=0.6, zorder=3)
    ax.scatter(x_all[is_int], y_all[is_int], s=2*s, color='blue', label="Interior", alpha=0.6, zorder=2)
    ax.scatter(x_all[is_data], y_all[is_data], s=s, color='black', marker='x', label="Measurements", linewidths=1, zorder=4)
    ax.set_xlabel(r"$\tilde x$")
    ax.set_ylabel(r"$\tilde y$")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


    # --------------------------------------------------------------------------
    # DATASET Grid collocation + bc
    # --------------------------------------------------------------------------
    pool_grid = init_pool(
        all_inputs_scaled=all_inputs_train_scaled,
        all_outputs_scaled=all_outputs_train_scaled,
        sensor_ids=all_ext_ids_train,
        R=R,
        x_max=x_max,
        all_inputs_grid_scaled=inputs_grid_scaled,
        mask_int_selected_grid=mask_int_selected_grid,
        mask_bc_selected_grid=mask_bc_selected_grid,
        tol=1e-4,
        device=device
    )

    dataset_grid = PINNDataset(pool_grid, include_int=True, include_bc=True, include_data=True, device=device)

    x_all, y_all = dataset_grid.inputs.cpu().numpy().T
    is_int = dataset_grid.is_int.cpu().numpy()
    is_bc = dataset_grid.is_bc.cpu().numpy()
    is_data = dataset_grid.is_data.cpu().numpy()

    ax = axes[1]  
    ax.plot(x_circle, y_circle, 'r-', linewidth=1)
    ax.plot(x_circle, y_circle, 'r-', linewidth=1, label="Tunnel wall", zorder=5)
    ax.scatter(x_all[is_bc], y_all[is_bc], s=2*s, color='darkgreen', alpha=0.6, zorder=3)
    ax.scatter(x_all[is_int], y_all[is_int], s=2*s, color='blue', alpha=0.6, zorder=2)
    ax.scatter(x_all[is_data], y_all[is_data], s=s, color='black', marker='x', linewidths=1, zorder=4)
    ax.set_xlabel(r"$\tilde x$")
    ax.set_ylabel(r"$\tilde y$")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


    # -------
    # Legend
    # -------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="black",
        bbox_to_anchor=(0.5, -0.05)
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(os.path.join(save_path, "measurements_vs_grid.pdf"), bbox_inches="tight", dpi=300)

    plt.show()




def plot_preds_2d(model, x_max, u_a, R, save_dir=None, prefix=None, device=torch.device("cpu")):
    
    # Define grid
    N = 200
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)


    model.eval()
    xy = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        ux_pred, uy_pred = model(xy)
        ux_pred, uy_pred = -ux_pred.cpu().numpy()*u_a.cpu().numpy(), -uy_pred.cpu().numpy()*u_a.cpu().numpy()

    ux_pred = ux_pred.reshape(N, N)
    uy_pred = uy_pred.reshape(N, N)

    # Mask for the tunnel
    X_scaled, Y_scaled = X*x_max, Y*x_max
    mask = X_scaled**2 + Y_scaled**2 <= R**2
    ux_pred = np.where(mask, np.nan, ux_pred)
    uy_pred = np.where(mask, np.nan, uy_pred)

    # --- ux ---
    fig, ax = plt.subplots(figsize=(7,7))
    cf = ax.contourf(X_scaled, Y_scaled, ux_pred, levels=50, cmap="RdBu_r", extend="both")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.3)
    plt.colorbar(cf, cax=cax, label=r"$u_x$")

    circle = plt.Circle((0,0), R, color="red", fill=False, linewidth=3, linestyle="-")
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"ux{('_' + prefix) if prefix else ''}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- uy ---
    fig, ax = plt.subplots(figsize=(7,7))
    cf = ax.contourf(X_scaled, Y_scaled, uy_pred, levels=50, cmap="RdBu_r", extend="both")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.3)
    plt.colorbar(cf, cax=cax, label=r"$u_y$")

    circle = plt.Circle((0,0), R, color="red", fill=False, linewidth=3, linestyle="-")
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"uy{('_' + prefix) if prefix else ''}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def plot_relative_errors_all(Eh_err, 
                             Ev_err, 
                             Gvh_err, 
                             K_err, 
                             beta_err,
                             Eh_std=None,
                             Ev_std=None,
                             Gvh_std=None,
                             K_std=None,
                             beta_std=None,
                             save_dir=None, 
                             zoom_xlim=None,
                             zoom_ylim=(0, 30), 
                             show_figs=False):
    
    
    os.makedirs(save_dir, exist_ok=True)
    x_vals = np.arange(1, len(Eh_err) + 1)
    err_dict = {
        r'$E_h$': (Eh_err, Eh_std, 'Eh_err.pdf'),
        r'$E_v$': (Ev_err, Ev_std, 'Ev_err.pdf'),
        r'$G_{vh}$': (Gvh_err, Gvh_std, 'Gvh_err.pdf'),
        r'$K$': (K_err, K_std, 'K_err.pdf'),
        r'$\beta$': (beta_err, beta_std, 'beta_err.pdf'),
    }
    
    # Tracés individuels
    for label, (err, std, filename) in err_dict.items():
        if err is not None:
            
            plt.figure(figsize=(7, 5))
            plt.plot(x_vals, err, color='tab:red', label=f'Relative error {label}')
            
            if std is not None:
                plt.fill_between(x_vals,
                 (err - std),
                 (err + std),
                 color='tab:red', alpha=0.2)            
            
            plt.xlabel('Step')
            plt.ylabel(f'{label} (%)')
            plt.grid(True)
            plt.legend()
            plt.ylim(*zoom_ylim)
            if zoom_xlim is not None:
                plt.xlim(*zoom_xlim)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            if save_dir is not None:
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show_figs:
                plt.show()
            else:
                plt.close()
                
    # Zoom
    plt.figure(figsize=(10, 6))
    for label, (err, _, _) in err_dict.items():
        if err is not None:
            plt.plot(x_vals, err, label=label)
    plt.xlabel('Step')
    plt.ylabel('Relative error (%)')
    plt.grid(True)
    plt.legend()
    plt.ylim(*zoom_ylim)
    if zoom_xlim is not None:
        plt.xlim(*zoom_xlim)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'all_errors_zoom.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_figs:
        plt.show()
    else:
        plt.close()
        
        

def compute_relative_errors(results_dict, device=torch.device("cpu")):
    """
    Compute relative L2 error for displacement predictions (ux, uy) over test dataset.
    """
    ux_preds_list = results_dict['all_ux_preds']
    uy_preds_list = results_dict['all_uy_preds']
    test_dataset = results_dict['test_dataset']

    # True displacements
    u_true = torch.stack([test_dataset[i]['outputs'] for i in range(len(test_dataset))]).to(device)  

    rel_errors = []
    for ux_pred, uy_pred in zip(ux_preds_list, uy_preds_list):
        ux_pred = torch.tensor(ux_pred, device=device).flatten()
        uy_pred = torch.tensor(uy_pred, device=device).flatten()

        # Stack correctly along last dimension
        pred = torch.stack([ux_pred, uy_pred], dim=-1)  # (N, 2)

        # Compute relative L2 error
        rel_err = torch.norm(pred - u_true) / torch.norm(u_true)
        rel_errors.append(rel_err.item())

    return rel_errors
    


def compute_relative_errors_from_test_dataset(model, test_dataset, device=torch.device("cpu")):
    """
    Compute relative L2 error for displacement predictions (ux, uy) over test dataset.
    """
    inputs = test_dataset.inputs
    u_true = test_dataset.outputs

    model.eval()
    with torch.no_grad():
        ux_preds, uy_preds = model(inputs)

    # Concatenate preds
    u_preds = torch.stack([ux_preds, uy_preds], dim=-1)

    # Compute relative L2 error
    rel_err = torch.norm(u_preds - u_true) / torch.norm(u_true)

    return rel_err.item()
    


def plot_curve(x, y, xlabel, ylabel, save_path=None, ylim=None, legend=None, marker=None):
    plt.figure(figsize=(8,6))
    plt.plot(x, y, marker=marker, label=legend)
    if legend:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if ylim:
        plt.ylim(*ylim)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    


def plot_relative_test_error(results_dict, save_dir):
    rel_errors = compute_relative_errors(results_dict)
    plot_curve(
        range(1, len(rel_errors)+1),
        np.array(rel_errors)*100,
        xlabel='Number of iterations',
        ylabel='Relative test error (%)',
        save_path=f'{save_dir}/relative_test_error.pdf',
        ylim=(0, 30),
        marker='o'
        
    )    


def plot_r2(results_dict, save_dir):
    R2_ux = results_dict['all_R2_ux']
    R2_uy = results_dict['all_R2_uy']
    
    n_iterations = len(results_dict['all_R2_ux'])
    
    x = np.arange(1, n_iterations + 1)
    
    plt.figure(figsize=(8,6))
    plt.plot(x, R2_ux, label=r'$u_x$')
    plt.plot(x, R2_uy, label=r'$u_y$')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$R^2$')
    plt.grid()
    plt.ylim(0.8,1)
    
    if save_dir:
        save_path = os.path.join(save_dir, "r2")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    


def plot_losses(results_dict, save_dir):
    # Train loss
    train_total_loss = [val for cycle in results_dict['train_total_loss'] for val in cycle]
    plot_curve(range(len(train_total_loss)), train_total_loss,
               xlabel='Global iteration', ylabel='Train total loss',
               save_path=f'{save_dir}/train_loss_evolution.pdf' if save_dir is not None else None)

    # Test loss
    test_total_loss = results_dict['test_total_loss']
    plot_curve(range(len(test_total_loss)), test_total_loss,
               xlabel='Global iteration', ylabel='Test total loss',
               save_path=f'{save_dir}/test_loss_evolution.pdf')
    


def plot_parameters(results_dict, save_dir):
    for param in ['Eh', 'Ev', 'Gvh', 'K']:
        values = results_dict[f'{param}_opt']
        flat = [val for cycle in values for val in cycle]
        plot_curve(range(len(flat)), flat,
                   xlabel='Global iteration', ylabel=f'${param}$ optimized',
                   save_path=f'{save_dir}/{param}_evolution.pdf')


def compute_relative_errors_from_dict(results_dict, device=torch.device("cpu")):

    ux_preds_list = results_dict['all_ux_preds']
    uy_preds_list = results_dict['all_uy_preds']
    test_dataset = results_dict['test_dataset']

    # True displacements
    u_true = torch.stack([test_dataset[i]['outputs'] for i in range(len(test_dataset))]).to(device)  

    rel_errors = []
    for ux_pred, uy_pred in zip(ux_preds_list, uy_preds_list):
        ux_pred = torch.tensor(ux_pred, device=device).flatten()
        uy_pred = torch.tensor(uy_pred, device=device).flatten()

        # Stack correctly along last dimension
        pred = torch.stack([ux_pred, uy_pred], dim=-1)  # (N, 2)

        # Compute relative L2 error
        rel_err = torch.norm(pred - u_true) / torch.norm(u_true)
        rel_errors.append(rel_err.item())

    return rel_errors
    