import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from losses import pinn_loss
from visualization import (
    plot_training_diagnostics,
    plot_predictions_vs_true,
    plot_preds_2d_MC,
    relative_error,
    compute_relative_errors_from_test_dataset
)


def train_pinn(model, 
               train_loader, 
               scale, 
               nadam_epochs=0, 
               es=False,
               checkpoint_interval=None, 
               checkpoint_name=None):

    (train_total_loss, train_loss_eq1, train_loss_eq2, train_loss_eq3, train_loss_eq4, 
     train_loss_ux, train_loss_uy)  = [], [], [], [], [], [], []
    Eh_opt, Ev_opt, Gvh_opt, K_opt, beta_opt = [], [], [], [], []
    
    batch_size = train_loader.batch_size   
    
    start_time = time.time()
    
    if nadam_epochs > 0 :
        
        nadam_optimizer = torch.optim.NAdam(
            list(model.ux_params.parameters()) + list(model.uy_params.parameters()) +
            [model.Eh_normalized, model.Ev_normalized, model.Gvh_normalized, model.K, model.beta_],
            lr=1e-4
        )

        # Scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(nadam_optimizer, 
        #                                                        mode='min', 
        #                                                        factor=0.8, 
        #                                                        patience=2000, 
        #                                                        threshold=1e-2, 
        #                                                        threshold_mode='rel',
        #                                                        min_lr=1e-5,
        #                                                        )
        
        # Early Stopping
        def get_patience(epoch):
            p0 = 1.5e4
            p_min = 500
            r = 1.5e-4 
            return int(max(p_min, p0*np.exp(-r*epoch)))
        
        threshold_rel = 1e-2 
        best_loss = np.inf
        epochs_no_improve = 0

        
        print("Nadam stage")  
        for epoch in range(1, nadam_epochs+1):
            
            # Training mode
            model.train()   
            (epoch_total_loss, epoch_loss_eq1, epoch_loss_eq2, epoch_loss_eq3 , epoch_loss_eq4,
             epoch_loss_ux, epoch_loss_uy) = 0, 0, 0, 0, 0, 0, 0
            
            for i, batch in enumerate(train_loader):
                
                nadam_optimizer.zero_grad()
                total_loss, loss_eq1, loss_eq2, loss_eq3, loss_eq4, loss_ux, loss_uy = pinn_loss(model, batch)
                total_loss.backward(retain_graph=True)
                nadam_optimizer.step()
    
                epoch_total_loss += total_loss.item()
                epoch_loss_eq1 += loss_eq1.item()
                epoch_loss_eq2 += loss_eq2.item()
                epoch_loss_eq3 += loss_eq3.item()
                epoch_loss_eq4 += loss_eq4.item()
                epoch_loss_ux += loss_ux.item()
                epoch_loss_uy += loss_uy.item()
        
            # Following the evolution of loss terms
            epoch_total_loss /= len(train_loader)
            train_total_loss.append(epoch_total_loss)
            train_loss_eq1.append(epoch_loss_eq1/len(train_loader))
            train_loss_eq2.append(epoch_loss_eq2/len(train_loader))
            train_loss_eq3.append(epoch_loss_eq3/len(train_loader))
            train_loss_eq4.append(epoch_loss_eq4/len(train_loader))
            train_loss_ux.append(epoch_loss_ux/len(train_loader))
            train_loss_uy.append(epoch_loss_uy/len(train_loader))

            # if scheduler applies
            # scheduler.step(epoch_total_loss)                
        
            # Following the evolution of PDE parameters        
            Eh_opt_ = model.Eh_normalized.item()
            Ev_opt_ = model.Ev_normalized.item()
            Gvh_opt_ = model.Gvh_normalized.item()
            K_opt_ = model.K.item()
            beta_opt_ = model.beta_.item()

            Eh_opt.append(Eh_opt_)
            Ev_opt.append(Ev_opt_)
            Gvh_opt.append(Gvh_opt_)
            K_opt.append(K_opt_)
            beta_opt.append(beta_opt_)

            early_stopping_patience = get_patience(epoch)

            # Display loss every 100 epochs 
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:01d} | Loss: {total_loss.item():.2e} | "
                    f"Eh: {Eh_opt_/scale:.2e} | Ev: {Ev_opt_/scale:.2e} | "
                    f"Gvh: {Gvh_opt_/scale:.2e} | K: {K_opt_:.2f} | "
                    f"Beta: {beta_opt_*180/np.pi:.2f} "
                    f"lr: {nadam_optimizer.param_groups[0]['lr']:.1e} | "
                    f"Epochs_no_improve: {epochs_no_improve}"
                )

            # Save checkpoint at every checkpoint interval
            if checkpoint_interval:
                if epoch % checkpoint_interval == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': nadam_optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': total_loss.item(),
                        'Eh': model.Eh_normalized.item(),
                        'Gvh': model.Gvh_normalized.item(),
                        'Ev': model.Ev_normalized.item(),
                        'K': model.K.item(),
                    }, f"{checkpoint_name}.pth")
                    print(f"Checkpoint saved at epoch {epoch}")


            # If early stopping
            # early_stopping_patience = get_patience(epoch)
            if es:
                if epoch_total_loss < best_loss * (1 - threshold_rel):
                    best_loss = epoch_total_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
    
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    end_time = time.time()
    
    # Total time
    training_time = end_time - start_time
    print(f"Training time: {training_time:.1f} seconds")

    return (train_total_loss, train_loss_eq1, train_loss_eq2, train_loss_eq3, train_loss_eq4,
            train_loss_ux, train_loss_uy, Eh_opt, Ev_opt, Gvh_opt, K_opt, beta_opt)




def train_step(step, model, train_loader, Eh, Ev, Gvh, K, beta, b, 
               logs, save_step_dir):

    # Define epochs
    nadam_epochs = 30000
    
    # Train
    es = True if step>=2 else False
    (train_total_loss, train_loss_eq1, train_loss_eq2, train_loss_eq3, train_loss_eq4, train_loss_ux, train_loss_uy, 
     Eh_opt, Ev_opt, Gvh_opt, K_opt, beta_opt) = train_pinn(model, train_loader, b,
                                                  es=es,
                                                  nadam_epochs=nadam_epochs, 
                                                 )

    # Save model states
    save_path = os.path.join(save_step_dir, f"step_{step}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)

    # Store logs (even NaNs)
    logs["all_train_total_loss"].append(train_total_loss)
    logs["all_train_loss_eq1"].append(train_loss_eq1)
    logs["all_train_loss_eq2"].append(train_loss_eq2)
    logs["all_train_loss_eq3"].append(train_loss_eq3)
    logs["all_train_loss_eq4"].append(train_loss_eq4)
    logs["all_train_loss_ux"].append(train_loss_ux)
    logs["all_train_loss_uy"].append(train_loss_uy)  

    logs["all_Eh_opt"].append(Eh_opt) 
    logs["all_Ev_opt"].append(Ev_opt) 
    logs["all_Gvh_opt"].append(Gvh_opt) 
    logs["all_K_opt"].append(K_opt) 
    logs["all_beta_opt"].append(beta_opt) 

    logs["Eh_err"].append(relative_error(Eh_opt[-1]/b, Eh)) 
    logs["Ev_err"].append(relative_error(Ev_opt[-1]/b, Ev))
    logs["Gvh_err"].append(relative_error(Gvh_opt[-1]/b, Gvh))
    logs["K_err"].append(relative_error(K_opt[-1], K))
    logs["beta_err"].append(relative_error(beta_opt[-1], beta))
    


    # To further exit loop if NaNs before chaos
    params = [Eh_opt, Ev_opt, Gvh_opt, K_opt, beta_opt]
    if any(torch.isnan(torch.tensor(p)).any().item() for p in params):
        print(f"NaN in parameters at step {step}, stopping training loop.")

        nan_keys = ["all_train_total_loss", 
                    "all_train_loss_eq1", "all_train_loss_eq2", "all_train_loss_eq3", "all_train_loss_eq4",
                   "all_train_loss_ux", "all_train_loss_uy",
                   "all_Eh_opt", "all_Ev_opt", "all_Gvh_opt", "all_K_opt", "all_beta_opt",
                   "Eh_err", "Ev_err", "Gvh_err", "K_err", "beta_err"]

        for key in nan_keys:
            logs[key].append(float('nan'))
        return False

    
    # Plot training curves
    plot_training_diagnostics(train_total_loss, train_loss_eq1, train_loss_eq2,
                                   train_loss_eq3, train_loss_eq4, train_loss_ux, train_loss_uy, 
                                   nadam_epochs=len(train_total_loss), scale=b,
                                   Eh_opt=Eh_opt, Ev_opt=Ev_opt, Gvh_opt=Gvh_opt, K_opt=K_opt, beta_opt=beta_opt,
                                   Eh_true=Eh, Ev_true=Ev, Gvh_true=Gvh, K_true=K, beta_true=beta,
                                   show_error=True,
                                   plot_losses=True,   
                                   plot_params=True,
                                   save_path=save_step_dir)
    return True


def evaluate_model(model, test_dataset, pool, R, x_max, u_a, logs, save_step_dir, n_samples=50, device=torch.device("cpu")):
    """
    Evaluate a PINN model with Monte Carlo Dropout.
    This performs multiple stochastic forward passes with dropout active at test time
    to estimate epistemic uncertainty and averaged predictions.
    """
    model.train()  
    inputs_test_scaled = test_dataset.inputs

    ux_preds = []
    uy_preds = []

    # --- Monte Carlo sampling ---
    for _ in range(n_samples):
        with torch.no_grad():
            ux_pred, uy_pred = model(inputs_test_scaled)
        ux_preds.append(ux_pred.unsqueeze(0))
        uy_preds.append(uy_pred.unsqueeze(0))

    ux_preds = torch.cat(ux_preds, dim=0)
    uy_preds = torch.cat(uy_preds, dim=0)

    # --- Mean and std across samples ---
    ux_mean = ux_preds.mean(dim=0)
    uy_mean = uy_preds.mean(dim=0)
    ux_std = ux_preds.std(dim=0)
    uy_std = uy_preds.std(dim=0)

    # --- Plot mean predictions vs true ---
    r2 = plot_predictions_vs_true(model=model,
                                  test_dataset=test_dataset,
                                  save_dir=None, 
                                  n_samples=50, 
                                  device=device)
    
    logs["all_R2_ux"].append(r2[0])
    logs["all_R2_uy"].append(r2[1])
    logs["all_ux_preds"].append(ux_mean)
    logs["all_uy_preds"].append(uy_mean)
    logs["all_ux_stds"].append(ux_std)
    logs["all_uy_stds"].append(uy_std)

    # --- Compute averaged losses ---
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    test_batch = next(iter(test_loader))

    total_losses = torch.zeros(7, device=device)
    for _ in range(n_samples):
        losses = pinn_loss(model, test_batch, data=True, residuals=True, test=True)
        total_losses += torch.tensor([l.item() for l in losses], device=device)
    avg_losses = total_losses / n_samples

    loss_names = [
        "test_total_loss", "test_loss_eq1", "test_loss_eq2",
        "test_loss_eq3", "test_loss_eq4", "test_loss_ux", "test_loss_uy"
    ]
    for name, val in zip(loss_names, avg_losses):
        logs[name].append(val.item())

    # --- Relative error on mean prediction ---
    logs["rel_test_errors"].append(
        compute_relative_errors_from_test_dataset(model, test_dataset, device=device)
    )

    # --- Visualization with MC uncertainty ---
    plot_preds_2d_MC(model, x_max, u_a, R, save_dir=save_step_dir,
                     prefix=None, n_samples=n_samples, device=device)

    return logs