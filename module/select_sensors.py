import torch
from pool_utils import update_pool

def ids_sensor(pool, sensor_type, sensor_id):

    if sensor_type not in ["extensometer", "convergence"]:
        raise ValueError(
            f"Invalid sensor_type: '{sensor_type}'. "
            "Expected 'extensometer' or 'convergence'." )
    
    masks = pool["masks"]
    if sensor_type == 'extensometer':
        # available points
        available_mask = (
            (pool["sensor_ids"] == sensor_id) &
             (pool["masks"]["is_data_available"])
        )

        candidate_ids = torch.where(available_mask)[0]

        # Separate 
        int_ids = candidate_ids[masks["is_int_available"][candidate_ids]]
        bc_ids = candidate_ids[masks["is_bc_available"][candidate_ids]]
        data_ids = candidate_ids[masks["is_data_available"][candidate_ids]]
        sensor_ids = candidate_ids 
        
    elif sensor_type == 'convergence':

        available_mask = (
            (pool["sensor_ids"] == sensor_id)
            & (
                (masks["is_bc_available"] & masks["is_data_available"])
                | (masks["is_bc_selected"] & masks["is_data_available"])
            )
        )

        candidate_ids = torch.where(available_mask)[0]
        
        # Separate
        int_ids = None
        bc_ids = candidate_ids[masks["is_bc_available"][candidate_ids]]
        data_ids = candidate_ids[masks["is_data_available"][candidate_ids]]

        # Add sensor
        unique_sensor_ids = torch.unique(pool["sensor_ids"][candidate_ids])
        mask_all_same_sensor = torch.isin(pool["sensor_ids"], unique_sensor_ids)
        all_indices_same_sensor = torch.where(mask_all_same_sensor)[0]
        sensor_ids = all_indices_same_sensor


    return int_ids, bc_ids, data_ids, sensor_ids

        

def uncertainty_sensor(model, pool, sensor_type, sensor_id, n_MC=50, device=torch.device("cpu")):

    if sensor_type not in ["extensometer", "convergence"]:
        raise ValueError(
            f"Invalid sensor_type: '{sensor_type}'. "
            "Expected 'extensometer' or 'convergence'." )

    masks = pool["masks"]
    if sensor_type == 'extensometer':
        available_mask = (
            (pool["sensor_ids"] == sensor_id) &
            (masks["is_data_available"])
        )
        candidate_ids = torch.where(available_mask)[0]

    elif sensor_type == 'convergence':
        available_mask = (
                (pool["sensor_ids"] == sensor_id)
                & (
                    (masks["is_bc_available"] & masks["is_data_available"])
                    | (masks["is_bc_selected"] & masks["is_data_available"])
                )
            )
        candidate_ids = torch.where(available_mask)[0]

    if len(candidate_ids) == 0:
        return -1, None
        
        
    # MC Dropout for epistemic uncertainty
    inputs = pool["inputs"][candidate_ids].to(device)
    ux_preds, uy_preds = [], []
    for _ in range(n_MC):
        ux_pred, uy_pred = model(inputs)
        ux_preds.append(ux_pred.detach())
        uy_preds.append(uy_pred.detach())
    ux_preds = torch.stack(ux_preds, dim=0)
    uy_preds = torch.stack(uy_preds, dim=0)
    ux_var = ux_preds.var(dim=0)
    uy_var = uy_preds.var(dim=0)
    total_uncertainty = (ux_var + uy_var).mean().item()

    return total_uncertainty, candidate_ids
    
    

def add_sensor_pool(model, 
                    pool, 
                    sensor_type='extensometer', 
                    sensor_ids_list=None, 
                    n_MC=50, 
                    random=False, 
                    device=torch.device("cpu")):

    """
    Add sensors to the pool either from a given list or by selecting the sensor
    with the highest epistemic uncertainty (or randomly if `random=True`).
    """

    # Check correct sensor type
    if sensor_type not in ["extensometer", "convergence"]:
        raise ValueError(
            f"Invalid sensor_type: '{sensor_type}'. "
            "Expected 'extensometer' or 'convergence'."
        )

    masks = pool["masks"]
    if sensor_ids_list is not None:
        for sensor_id in sensor_ids_list:
    
            # Check sensor availability
            if not torch.any(masks["is_sensor_available"][pool["sensor_ids"] == sensor_id]):
                continue  


            int_ids, bc_ids, data_ids, sensor_ids = ids_sensor(pool=pool, 
                                                               sensor_type=sensor_type, 
                                                               sensor_id=sensor_id) 
            # Update pool
            pool = update_pool(
                pool,
                new_int_ids=int_ids,
                new_bc_ids=bc_ids,
                new_data_ids=data_ids,
                new_sensor_ids=sensor_ids  
            )
    
        return pool


    else :
        model.train() 
        sensor_ids_available = torch.unique(pool["sensor_ids"][masks["is_sensor_available"]])

        if len(sensor_ids_available) == 0:
            print("No available sensors in the pool.")
            return pool
        
        if random : 
            best_sensor_id = sensor_ids_available[torch.randint(len(sensor_ids_available), (1,)).item()]
        
        else:
            uncertainties = []
            for sensor_id in sensor_ids_available:

                total_uncertainty, candidate_ids = uncertainty_sensor(model=model, 
                                                                 pool=pool, 
                                                                 sensor_type=sensor_type,
                                                                 sensor_id=sensor_id,
                                                                 n_MC=50, 
                                                                 device=device)

                uncertainties.append(total_uncertainty)
        
            # Tensor conversion
            uncertainties = torch.tensor(uncertainties)
            if torch.all(uncertainties < 0):
                print("No available sensors with data for uncertainty estimation.")
                return pool
            
            # Sensor selection
            best_sensor_pos = torch.argmax(uncertainties).item()
            best_sensor_id = sensor_ids_available[best_sensor_pos]

        # Add the selected sensor
        int_ids, bc_ids, data_ids, sensor_ids = ids_sensor(
            pool=pool,
            sensor_type=sensor_type,
            sensor_id=best_sensor_id
        )
    
        pool = update_pool(
            pool=pool,
            new_int_ids=int_ids,
            new_bc_ids=bc_ids,
            new_data_ids=data_ids,
            new_sensor_ids=sensor_ids
        )
    
        print(
            f"Capteur {best_sensor_id} added — "
            f"{'mean uncertainty = {:.2e}'.format(uncertainties[best_sensor_pos]) if not random else ''} — "
            f"{len(data_ids)} points added"
        )

        return pool