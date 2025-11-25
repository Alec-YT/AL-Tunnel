import os
import pickle
import torch
from torch.utils.data import DataLoader
from pool_utils import initialize_train_pool, add_full_grid, test_dataset_from_true_data, PINNDataset, compute_scaling_factor
from select_sensors import add_sensor_pool
from visualization import plot_dataset_points
from train import train_step, evaluate_model

def active_training(model, 
                    sensor_type,
                    n_sensors, 
                    R, 
                    L, 
                    sigma_v, 
                    Eh, 
                    Ev, 
                    Gvh, 
                    K, 
                    beta, 
                    noise_level, 
                    base_dir, 
                    save_dir, 
                    random=False,
                    device=torch.device("cpu")):
    
    # Logs
    logs = {
        "Eh_err": [], "Ev_err": [], "Gvh_err": [], "K_err": [], "beta_err": [],
        "all_Eh_opt": [], "all_Ev_opt": [], "all_Gvh_opt": [], "all_K_opt": [], "all_beta_opt": [], 
        "all_train_total_loss": [], "all_train_loss_eq1": [], "all_train_loss_eq2": [],
        "all_train_loss_eq3": [], "all_train_loss_eq4": [], "all_train_loss_ux": [], "all_train_loss_uy": [],
        "test_total_loss": [], "test_loss_eq1": [], "test_loss_eq2": [],
        "test_loss_eq3": [], "test_loss_eq4": [], "test_loss_ux": [], "test_loss_uy": [],
        "rel_test_errors" : [],
        "all_R2_ux": [], "all_R2_uy": [],
        "scale": [],
        "all_ux_preds": [], "all_uy_preds": [],
        "all_ux_stds": [], "all_uy_stds": []
    }

    # Initialize and save current states
    step = 0
    n_sensors_added = 0

    # Sub step folder
    save_step_dir = os.path.join(save_dir, f'step_{step}')
    os.makedirs(save_step_dir, exist_ok=True)
    save_path = os.path.join(save_step_dir, f"step_{step}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)

    
    # -----------------------------
    # Initialize train pool
    # -----------------------------
    train_pool, x_max, u_a = initialize_train_pool(base_dir, 
                                                        sigma_v, R, L,
                                                        selected_ids=None,
                                                        noise_level=noise_level,
                                                        device=device, 
                                                        )

    # -----------------------------
    # Initialize test data
    # -----------------------------
    test_dataset, *_ = test_dataset_from_true_data(R, L, device=device)
    logs["test_dataset"] = test_dataset

    # --- Scaling factor --- 
    b = compute_scaling_factor(sigma_v, x_max, u_a)
    
    # --------------------
    # Main Loop
    # --------------------
    while n_sensors_added < n_sensors :

        # --------------------
        # 1. Add sensor 
        # --------------------
        # New step
        step+=1

        if step == 1: # initialization
            sensor_ids_list = [10, 19]
            n_sensors_added += 2
            sensor_type_init = 'extensometer'
        else :
            sensor_ids_list = None
            n_sensors_added += 1
        

        # New sub step folder
        save_step_dir = os.path.join(save_dir, f'step_{step}')
        os.makedirs(save_step_dir, exist_ok=True)

        train_pool = add_sensor_pool(model, 
                                     pool=train_pool, 
                                     sensor_type=sensor_type_init if step == 1 else sensor_type, 
                                     sensor_ids_list=sensor_ids_list, 
                                     n_MC=50, 
                                     random=random, 
                                     device=device)
      
        
        # Generate dataset
        train_dataset = PINNDataset(train_pool, include_int=True, include_bc=True, include_data=True, device=device)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        
        # Plot dataset
        plot_dataset_points(train_dataset, R, x_max, save_path=save_step_dir)


        # Train step
        if not train_step(step=step, 
                   model=model, 
                   train_loader=train_loader, 
                   Eh=Eh, 
                   Ev=Ev, 
                   Gvh=Gvh, 
                   K=K,
                   beta=beta,
                   b=b, 
                   logs=logs, 
                   save_step_dir=save_step_dir):
            break 
            
        # Evaluate model 
        logs = evaluate_model(model=model, 
                       test_dataset=test_dataset, 
                       pool=train_pool,
                       R=R,
                       x_max=x_max,
                       u_a=u_a,
                       logs=logs, 
                       save_step_dir=save_step_dir, 
                       device=device)

    
        # -----------------------------------
        # Add the full grid - Ends initialization
        # -----------------------------------
        if step == 1 :


            train_pool = add_full_grid(model=model, 
                                       pool=train_pool, 
                                       device=device)

    
            # New step
            step+=1
    
            # New sub step folder
            save_step_dir = os.path.join(save_dir, f'step_{step}')
            os.makedirs(save_step_dir, exist_ok=True)
            
            # New DataLoader
            train_dataset = PINNDataset(train_pool, device=device, include_int=True, include_bc=True, include_data=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
            
            # Plot
            plot_dataset_points(train_dataset, R, x_max, save_path=save_step_dir)
    
            # Train step
            if not train_step(step=step, 
                       model=model, 
                       train_loader=train_loader, 
                       Eh=Eh, 
                       Ev=Ev, 
                       Gvh=Gvh, 
                       K=K,
                       beta=beta,
                       b=b, 
                       logs=logs, 
                       save_step_dir=save_step_dir):
                break 
    
            # Evaluate model 
            logs = evaluate_model(model=model, 
                           test_dataset=test_dataset, 
                           pool=train_pool,
                           R=R,
                           x_max=x_max,
                           u_a=u_a,
                           logs=logs, 
                           save_step_dir=save_step_dir, 
                           device=device)

    # save results
    results_dict = {
        'Eh_err': logs["Eh_err"],
        'Ev_err': logs["Ev_err"],
        'Gvh_err': logs["Gvh_err"],
        'K_err': logs["K_err"],
        'beta_err': logs["beta_err"],
        'Eh_opt': logs["all_Eh_opt"],
        'Ev_opt': logs["all_Ev_opt"],
        'Gvh_opt': logs["all_Gvh_opt"],
        'K_opt': logs["all_K_opt"],
        'beta_opt': logs["all_beta_opt"],
        'all_R2_ux': logs["all_R2_ux"],
        'all_R2_uy': logs["all_R2_uy"],
        'train_total_loss': logs["all_train_total_loss"],
        'train_loss_eq1': logs["all_train_loss_eq1"],
        'train_loss_eq2': logs["all_train_loss_eq2"],
        'train_loss_eq3': logs["all_train_loss_eq3"],
        'train_loss_eq4': logs["all_train_loss_eq4"],
        'train_loss_ux': logs["all_train_loss_ux"],
        'train_loss_uy': logs["all_train_loss_uy"],
        'test_total_loss': logs["test_total_loss"],
        'test_loss_eq1': logs["test_loss_eq1"],
        'test_loss_eq2': logs["test_loss_eq2"],
        'test_loss_eq3': logs["test_loss_eq3"],
        'test_loss_eq4': logs["test_loss_eq4"],
        'test_loss_ux': logs["test_loss_ux"],
        'test_loss_uy': logs["test_loss_uy"],
        'rel_test_errors': logs["rel_test_errors"],
        'test_dataset': logs["test_dataset"],
        'all_ux_preds': logs["all_ux_preds"],
        'all_uy_preds': logs["all_uy_preds"],
        "all_ux_stds": logs["all_ux_stds"],
        "all_uy_stds": logs["all_uy_stds"],
        'scale': b,
        'n_sensors' : n_sensors,
    }

    save_path = os.path.join(save_dir, "all_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Results saved in {save_path}")

    return results_dict
    