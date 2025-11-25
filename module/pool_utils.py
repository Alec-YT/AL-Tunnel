import os
import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import griddata
from pathlib import Path


def split_inputs_outputs(df, device=torch.device("cpu")):
    # Extract inputs and outputs from DataFrame
    inputs = df[['x', 'y']].values
    outputs = - df[['disp_x', 'disp_y']].values  # Negative for compression
    
    # Convert to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    outputs = torch.tensor(outputs, dtype=torch.float32, device=device)
        
    return inputs, outputs
    

def train_test_files(base_dir, selected_ids=None):
    if selected_ids is not None:
        # Explicit selection
        return [
            (os.path.join(base_dir, f"TransverseIsotropyShear_out_real_extensometer_v2_{i}_0001.csv"), i)
            for i in selected_ids
        ]
    
    # Default: all files
    return [
        (os.path.join(base_dir, f"TransverseIsotropyShear_out_real_extensometer_v2_{i}_0001.csv"), i)
        for i in range(1, 37)
    ]


def load_all_extensometer_data(files):
    dfs = []
    for file, ext_id in files:
        df = pd.read_csv(file)
        df["extensometer_id"] = ext_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def select_inputs_outputs(files, noise_level=None, device=torch.device("cpu")):
    # --- Load data ---
    df = load_all_extensometer_data(files)
    inputs, outputs = split_inputs_outputs(df, device)

    if noise_level is not None:
        noise = noise_level * torch.randn_like(outputs) * torch.abs(outputs)
        outputs = outputs + noise

    # --- Unique ID for each point ---
    N = len(inputs)
    unique_ids = torch.arange(N, device=device)

    # --- Extensometer ID ---
    ext_ids = torch.tensor(df["extensometer_id"].values, device=device)

    return inputs, outputs, unique_ids, ext_ids
    

def normalize(inputs, outputs=None, R=None, L=None, device=torch.device("cpu")):
    x_max = R + L
    inputs_scaled = inputs / x_max
    inputs_scaled = inputs_scaled.to(device)

    u_a = torch.tensor([1e-1], dtype=torch.float32, device=device)
    outputs_scaled = None
    if outputs is not None:
        outputs_scaled = outputs / u_a
        outputs_scaled = outputs_scaled.to(device)

    return inputs_scaled, outputs_scaled, x_max, u_a


def compute_scaling_factor(sigma_v, x_max, u_a):
    return (1.0 / sigma_v * u_a / x_max).item()



class PINNDataset(Dataset):
    def __init__(self, pool, device=torch.device("cpu"),
                 include_int=True, include_bc=True, include_data=True, include_all=False):
        """
        pool : dict containing inputs, outputs, masks, sensor_ids
        include_* : booleans to select types of points for training
        include_all : if True, include all points (for test)
        """
        self.device = device
        masks = pool['masks']

        if include_all:
            # select all points
            selected_mask = torch.ones(len(pool['inputs']), dtype=torch.bool, device=device)
            self.is_int = (masks['is_int_selected'] | masks['is_int_available'])[selected_mask].to(device) 
            self.is_bc  = (masks['is_bc_selected']  | masks['is_bc_available'])[selected_mask].to(device)
            self.is_data = (masks['is_data_selected'] | masks['is_data_available'])[selected_mask].to(device)
        else:
            # select only the points flagged for training
            selected_mask = torch.zeros(len(pool['inputs']), dtype=torch.bool, device=device)
            if include_int:
                selected_mask |= masks['is_int_selected']
            if include_bc:
                selected_mask |= masks['is_bc_selected']
            if include_data:
                selected_mask |= masks['is_data_selected']

            self.is_int = masks['is_int_selected'][selected_mask].to(device)
            self.is_bc = masks['is_bc_selected'][selected_mask].to(device)
            self.is_data = masks['is_data_selected'][selected_mask].to(device)

        # Inputs, outputs, sensor ids
        self.inputs = pool['inputs'][selected_mask].to(device).requires_grad_(False)
        self.outputs = pool['outputs'][selected_mask].to(device).requires_grad_(False)
        self.ext_ids = pool['sensor_ids'][selected_mask].to(device)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'inputs': self.inputs[idx],
            'outputs': self.outputs[idx],
            'is_int': self.is_int[idx],
            'is_bc': self.is_bc[idx],
            'is_data': self.is_data[idx],
        }



def test_dataset_from_true_data(R, L, device=torch.device("cpu")):


    # Path to repo root
    current_file = Path(__file__).resolve()  
    repo_root = current_file.parents[1]      
    
    # Path to data file
    data_file = repo_root / "synthetic_data" / "TransverseIsotropyShear_out_point_sample_0001.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {data_file}")
        
    df = pd.read_csv(data_file)

    # Extract data
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    ux = -df["disp_x"].to_numpy()
    uy = -df["disp_y"].to_numpy()

    # --- Polar grid ---
    Nr, Ntheta = 50, 200  
    r = np.linspace(R, R + L, Nr)           
    theta = np.linspace(0, 2 * np.pi, Ntheta)
    Rg, TH = np.meshgrid(r, theta, indexing="ij")

    # Cartesian 
    Xg = Rg * np.cos(TH)
    Yg = Rg * np.sin(TH)

    # Interpolation 
    UX = griddata((x, y), ux, (Xg, Yg), method="linear")
    UY = griddata((x, y), uy, (Xg, Yg), method="linear")

    # Delete NaN
    mask_valid = ~np.isnan(UX) & ~np.isnan(UY)
    X = Xg[mask_valid]
    Y = Yg[mask_valid]
    UX = UX[mask_valid]
    UY = UY[mask_valid]

    # Tensor conversion
    inputs_test = torch.tensor(np.stack([X, Y], axis=1), dtype=torch.float32).to(device)
    outputs_test = torch.tensor(np.stack([UX, UY], axis=1), dtype=torch.float32).to(device)

    # Normalization
    inputs_test_scaled, outputs_test_scaled, x_max, u_a = normalize(
        inputs_test, outputs_test, R, L, device=device
    )

    # Pool and dataset
    test_pool = init_pool(
        all_inputs_scaled=inputs_test_scaled,
        all_outputs_scaled=outputs_test_scaled,
        sensor_ids=torch.full((len(inputs_test_scaled),), -1, device=device),
        R=R,
        x_max=x_max,
        device=device,
    )

    test_dataset = PINNDataset(test_pool, include_all=True, device=device)

    return test_dataset, x_max, u_a


def polar_grid_flat(
    R, 
    L, 
    Nr=10, 
    Ntheta=30, 
    refine_exponent=1.5, 
    select_every_r=2,
    select_every_theta=2,
    device=torch.device("cpu"),
):
    # --- Radial refinement ---
    r = R + L * (np.linspace(0, 1, Nr) ** refine_exponent)

    # --- Uniform Angular ---
    theta = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)

    # --- Full Grid ---
    r_flat = np.repeat(r, Ntheta)
    theta_flat = np.tile(theta, Nr)
    x_flat = r_flat * np.cos(theta_flat)
    y_flat = r_flat * np.sin(theta_flat)

    # --- Tensor conversion ---
    inputs = torch.tensor(np.stack([x_flat, y_flat], axis=1), dtype=torch.float32, device=device)
    

    # --- Initialized masks ---
    N = inputs.shape[0]
    mask_int_selected = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc_selected = torch.zeros(N, dtype=torch.bool, device=device)

    # --- Select indices ---
    selected_r_idx = np.arange(0, Nr, select_every_r)
    selected_theta_idx = np.arange(0, Ntheta, select_every_theta)

    for i_r in selected_r_idx:
        for i_theta in selected_theta_idx:
            idx = i_r * Ntheta + i_theta
            if np.isclose(r[i_r], R):  # BC points
                mask_bc_selected[idx] = True
            else:  # Int points
                mask_int_selected[idx] = True

    return inputs, mask_int_selected, mask_bc_selected


def init_pool(
    all_inputs_scaled: torch.Tensor,
    all_outputs_scaled: torch.Tensor,
    sensor_ids: torch.Tensor,
    R,
    x_max,
    all_inputs_grid_scaled: torch.Tensor = None,
    mask_int_selected_grid: torch.Tensor = None,
    mask_bc_selected_grid: torch.Tensor = None,
    tol=1e-3,
    device=torch.device("cpu")
):
    # --- Empty tensors if None---
    if all_inputs_grid_scaled is None:
        all_inputs_grid_scaled = torch.empty((0, all_inputs_scaled.shape[1]), device=device)

    if mask_int_selected_grid is None:
        mask_int_selected_grid = torch.zeros(all_inputs_grid_scaled.shape[0], dtype=torch.bool, device=device)

    if mask_bc_selected_grid is None:
        mask_bc_selected_grid = torch.zeros(all_inputs_grid_scaled.shape[0], dtype=torch.bool, device=device)

    # --- Dimensions ---
    N = len(all_inputs_scaled)
    N_new = len(all_inputs_grid_scaled)

    # --- Concat inputs ---
    all_inputs_scaled = torch.cat([all_inputs_scaled, all_inputs_grid_scaled], dim=0)
    all_outputs_scaled = torch.cat(
        [all_outputs_scaled, torch.full((N_new, 2), float('nan'), device=device)],
        dim=0
    )
    sensor_ids = torch.cat(
        [sensor_ids, torch.full((N_new,), -1, device=device)],
        dim=0
    )

    r = torch.norm(all_inputs_scaled, dim=1)

    # --- Interior ---
    mask_int_available = (r >= R / x_max + tol)
    mask_int_selected = torch.cat([
        torch.zeros(N, dtype=torch.bool, device=device),
        mask_int_selected_grid
    ], dim=0)

    # --- Boundary ---
    mask_bc_available = (r >= (R / x_max - tol)) & (r <= (R / x_max + tol))
    mask_bc_selected = torch.cat([
        torch.zeros(N, dtype=torch.bool, device=device),
        mask_bc_selected_grid
    ], dim=0)

    # --- Data ---
    mask_data_available = torch.cat([
        torch.ones(N, dtype=torch.bool, device=device),
        torch.zeros(N_new, dtype=torch.bool, device=device)
    ], dim=0)
    mask_data_selected = torch.zeros(N + N_new, dtype=torch.bool, device=device)

    # --- Sensors ---
    mask_sensor_available = torch.cat([
        torch.ones(N, dtype=torch.bool, device=device),
        torch.zeros(N_new, dtype=torch.bool, device=device)
    ], dim=0)
    mask_sensor_selected = torch.zeros(N + N_new, dtype=torch.bool, device=device)

    # --- Final pool ---
    pool = {
        "inputs": all_inputs_scaled,
        "outputs": all_outputs_scaled,
        "sensor_ids": sensor_ids,
        "masks": {
            "is_bc_selected": mask_bc_selected,
            "is_bc_available": mask_bc_available,
            "is_int_selected": mask_int_selected,
            "is_int_available": mask_int_available,
            "is_data_selected": mask_data_selected,
            "is_data_available": mask_data_available,
            "is_sensor_selected": mask_sensor_selected,
            "is_sensor_available": mask_sensor_available
        },
    }

    return pool
    

def update_pool(
    pool: dict,
    new_int_ids=None,
    new_bc_ids=None,
    new_data_ids=None,
    new_sensor_ids=None,
    make_copy=True,  
):
    if make_copy:
        pool = copy.deepcopy(pool)

    device = pool["inputs"].device

    # --- Interior ---
    if new_int_ids is not None:
        new_int_ids = torch.as_tensor(new_int_ids, dtype=torch.long, device=device)
        pool["masks"]["is_int_selected"][new_int_ids] = True
        pool["masks"]["is_int_available"][new_int_ids] = False

    # --- Boundary ---
    if new_bc_ids is not None:
        new_bc_ids = torch.as_tensor(new_bc_ids, dtype=torch.long, device=device)
        pool["masks"]["is_bc_selected"][new_bc_ids] = True
        pool["masks"]["is_bc_available"][new_bc_ids] = False

    # --- Data ---
    if new_data_ids is not None:
        new_data_ids = torch.as_tensor(new_data_ids, dtype=torch.long, device=device)
        pool["masks"]["is_data_selected"][new_data_ids] = True
        pool["masks"]["is_data_available"][new_data_ids] = False

    # --- Sensor ---
    if new_sensor_ids is not None:
        new_sensor_ids = torch.as_tensor(new_sensor_ids, dtype=torch.long, device=device)
        pool["masks"]["is_sensor_selected"][new_sensor_ids] = True
        pool["masks"]["is_sensor_available"][new_sensor_ids] = False

    return pool



def add_full_grid(model, pool, device=torch.device("cpu")):
    
    '''Add to the pool all the collocation and boundary points sampled from the second grid G'''
    
    masks = pool['masks']
    mask_int_available = masks['is_int_available'] & ~masks['is_sensor_available'] & ~masks['is_sensor_selected']
    mask_bc_available  = masks['is_bc_available'] & ~masks['is_sensor_available'] & ~masks['is_sensor_selected']
    mask_available = mask_int_available | mask_bc_available
    

    # --- INT and BC indices ---
    indices_global = torch.where(mask_available)[0]
    
    if len(indices_global) == 0 :
        print("No available point")
        return pool


    # --- INT / BC distinction ---
    selected_int = indices_global[mask_int_available[indices_global]]
    selected_bc  = indices_global[mask_bc_available[indices_global]]


    # --- Update pool ----
    pool = update_pool(
        pool=pool,
        new_int_ids=selected_int,
        new_bc_ids=selected_bc,
        new_data_ids=None,
        new_sensor_ids=None,
        make_copy=True
    )

    return pool



def initialize_train_pool(base_dir, sigma_v, R, L, selected_ids=None, noise_level=None, device=torch.device("cpu")):
    
    # Select all
    train_files = train_test_files(base_dir, selected_ids=selected_ids)
    
    # Split inputs outputs train
    all_inputs_train, all_outputs_train, all_unique_ids_train, all_ext_ids_train = select_inputs_outputs(train_files, 
                                                                                                         noise_level=noise_level, 
                                                                                                         device=device)

    # --- Scale inputs / outputs ---
    all_inputs_train_scaled, all_outputs_train_scaled, x_max, u_a = normalize(all_inputs_train, 
                                                                              all_outputs_train, 
                                                                              R=R, 
                                                                              L=L, 
                                                                              device=device)

    # --- Pool --- 
    inputs_grid, mask_int_selected_grid, mask_bc_selected_grid = polar_grid_flat(
                                                                                R=R, 
                                                                                L=L, 
                                                                                Nr=10, 
                                                                                Ntheta=36, 
                                                                                refine_exponent=1, 
                                                                                select_every_r=1,
                                                                                select_every_theta=1,
                                                                                device=device,
                                                                            )
    
    inputs_grid_scaled, *_ = normalize(inputs_grid, None, R, L)
    
    # The pool combines measurements and grid points that will be queried subsequently
    train_pool = init_pool(
                                all_inputs_scaled=all_inputs_train_scaled,
                                all_outputs_scaled=all_outputs_train_scaled,
                                sensor_ids=all_ext_ids_train,
                                R=R,
                                x_max=x_max,
                                all_inputs_grid_scaled=inputs_grid_scaled,
                                mask_int_selected_grid=None, 
                                mask_bc_selected_grid=None,
                                tol=1e-4,
                                device=device
                            )


    return train_pool, x_max, u_a
    
