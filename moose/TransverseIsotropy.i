# lambda = 1
s0 = 5e6 # Pa
beta = 45 # deg 
K =  0.75 
L = 1

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  file = TheFullMesh.msh
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    strain = SMALL
    generate_output = 'stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy '
    add_variables = true
    block = 'rock'
    material_output_family = 'MONOMIAL'
    material_output_order = 'CONSTANT'
    eigenstrain_names = ini_stress 
  []
[]

[Functions]
    [sigma_v]
        type = ParsedFunction
        symbol_names = 's0 beta K L'
        symbol_values = ' ${s0} ${beta} ${K} ${L}'
        expression = s0/2*((1+K)+(1-K)*cos(2*beta*pi/180))*L  
    []
    [sigma_h]
        type = ParsedFunction
        symbol_names = 's0 beta K L'
        symbol_values = ' ${s0} ${beta} ${K} ${L}'
        expression = s0/2*((1+K)-(1-K)*cos(2*beta*pi/180))*L  
    []
    [-sigma_v]
        type = ParsedFunction
        symbol_names = 's0 beta K L'
        symbol_values = ' ${s0} ${beta} ${K} ${L}'
        expression = -s0/2*((1+K)+(1-K)*cos(2*beta*pi/180))*L  
    []
    [-sigma_h]
        type = ParsedFunction
        symbol_names = 's0 beta K L'
        symbol_values = ' ${s0} ${beta} ${K} ${L}'
        expression = -s0/2*((1+K)-(1-K)*cos(2*beta*pi/180))*L  
    []
    [tau_vh]
        type = ParsedFunction
        symbol_names = 's0 beta K L'
        symbol_values = ' ${s0} ${beta} ${K} ${L}'
        expression = s0/2*(1-K)*sin(2*beta*pi/180)*L 
    []
    [-tau_vh]
        type = ParsedFunction
        symbol_names = 's0 beta K L'
        symbol_values = ' ${s0} ${beta} ${K} ${L}'
        expression = -s0/2*(1-K)*sin(2*beta*pi/180)*L 
    []
  []

[BCs]
  [top_normal_stress]
    type = FunctionNeumannBC
    variable = disp_y	
    boundary = 'top'
    function = -sigma_v # Pa
  []
  [bottom_normal_stress]
    type = FunctionNeumannBC
    variable = disp_y	
    boundary = 'bottom'
    function = sigma_v # Pa
  []
  [top_shear_stress]
    type = FunctionNeumannBC
    variable = disp_x	
    boundary = 'top'
    function = tau_vh # Pa
  []
  [bottom_shear_stress]
    type = FunctionNeumannBC
    variable = disp_x	
    boundary = 'bottom'
    function = -tau_vh # Pa
  []
  [right_normal_stress]
    type = FunctionNeumannBC
    variable = disp_x
    boundary = 'right'
    function = -sigma_h # Pa
  []
  [left_normal_stress]
    type = FunctionNeumannBC
    variable = disp_x
    boundary = 'left'
    function = sigma_h # Pa
  []
  [right_shear_stress]
    type = FunctionNeumannBC
    variable = disp_y
    boundary = 'right'
    function = tau_vh # Pa
  []
  [left_shear_stress]
    type = FunctionNeumannBC
    variable = disp_y
    boundary = 'left'
    function = -tau_vh # Pa
  []
  [cavityPressure_x]
    type = Pressure
    boundary = 'wall'
    variable = disp_x
    factor = 0 # Pa
    # function = sigma_h # deconfinement # Pa
  []
  [cavityPressure_y]
    type = Pressure
    boundary = 'wall'
    variable = disp_y
    factor = 0 # Pa
    # function = sigma_v # deconfinement # Pa
  []
[]

[Materials]
[elasticity]
    type = ComputeElasticityTensor 
    fill_method = symmetric9
    # C_ijkl = '7141705659.63 3382880574.06 3109447595.11 6232701178.88 3382880574.06 7141705659.63 1630000000.0 2016129032.26 1630000000.0' # cvcf book
    C_ijkl = '647541887.13 81327160.49 93970458.55 357839506.17 81327160.49 647541887.13 200000000.0 276785714.29 200000000.0' # Vu 2013 St Martin
    block = 'rock'
  []
  [stress]
    type = ComputeLinearElasticStress
    block = 'rock'
  []
  [strain_from_initial_stress] # To initialize the displacement to zero based on in situ initial stress
    type = ComputeEigenstrainFromInitialStress
    initial_stress = '-sigma_h tau_vh 0 tau_vh -sigma_v 0 0 0 -sigma_h'
    eigenstrain_name = ini_stress
    block = 'rock'
  []
[]

[Preconditioning]
  [SMP]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Steady 
  solve_type = NEWTON
  l_abs_tol = 1e-50
  l_tol = 1e-50
  line_search = none
  nl_abs_tol = 1E-6
  nl_rel_tol = 1E-6
[]



[VectorPostprocessors]

  [collocation_and_boundary_points]
    type = PositionsFunctorValueSampler
    functors = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
    positions = 'all_elements'
    sort_by = 'x'
    execute_on = TIMESTEP_END # Default value
  []
    [real_extensometer_v2_1]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '5.0 0.0 0'
       end_point = '29.0 0.0 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_2]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '4.92403876506104 0.8682408883346516 0'
       end_point = '28.559424837354033 5.0357971523409795 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_3]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '4.698463103929543 1.7101007166283435 0'
       end_point = '27.251086002791343 9.918584156444393 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_4]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '4.330127018922194 2.4999999999999996 0'
       end_point = '25.114736709748723 14.499999999999998 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_5]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '3.83022221559489 3.2139380484326963 0'
       end_point = '22.215288850450364 18.640840680909637 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_6]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '3.2139380484326967 3.83022221559489 0'
       end_point = '18.64084068090964 22.215288850450364 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_7]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '2.5000000000000004 4.330127018922193 0'
       end_point = '14.500000000000004 25.11473670974872 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_8]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '1.7101007166283442 4.698463103929542 0'
       end_point = '9.918584156444396 27.25108600279134 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_9]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '0.8682408883346521 4.92403876506104 0'
       end_point = '5.035797152340982 28.559424837354033 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_10]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '3.061616997868383e-16 5.0 0'
       end_point = '1.7757378587636622e-15 29.0 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_11]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-0.8682408883346515 4.92403876506104 0'
       end_point = '-5.035797152340979 28.559424837354033 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_12]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-1.7101007166283435 4.698463103929543 0'
       end_point = '-9.918584156444393 27.251086002791343 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_13]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-2.499999999999999 4.330127018922194 0'
       end_point = '-14.499999999999995 25.114736709748723 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_14]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-3.2139380484326967 3.83022221559489 0'
       end_point = '-18.64084068090964 22.215288850450364 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_15]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-3.8302222155948895 3.2139380484326976 0'
       end_point = '-22.21528885045036 18.640840680909644 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_16]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-4.330127018922194 2.4999999999999996 0'
       end_point = '-25.114736709748723 14.499999999999998 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_17]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-4.698463103929542 1.7101007166283444 0'
       end_point = '-27.25108600279134 9.918584156444398 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_18]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-4.92403876506104 0.8682408883346514 0'
       end_point = '-28.559424837354033 5.035797152340978 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_19]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-5.0 6.123233995736766e-16 0'
       end_point = '-29.0 3.5514757175273244e-15 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_20]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-4.92403876506104 -0.8682408883346524 0'
       end_point = '-28.559424837354033 -5.035797152340984 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_21]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-4.698463103929543 -1.7101007166283433 0'
       end_point = '-27.251086002791343 -9.91858415644439 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_22]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-4.330127018922193 -2.5000000000000004 0'
       end_point = '-25.11473670974872 -14.500000000000004 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_23]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-3.83022221559489 -3.2139380484326963 0'
       end_point = '-22.215288850450364 -18.640840680909637 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_24]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-3.2139380484326976 -3.8302222155948895 0'
       end_point = '-18.640840680909644 -22.21528885045036 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_25]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-2.500000000000002 -4.330127018922192 0'
       end_point = '-14.500000000000012 -25.114736709748712 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_26]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-1.7101007166283468 -4.698463103929541 0'
       end_point = '-9.918584156444412 -27.25108600279134 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_27]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-0.8682408883346516 -4.92403876506104 0'
       end_point = '-5.0357971523409795 -28.559424837354033 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_28]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '-9.184850993605148e-16 -5.0 0'
       end_point = '-5.327213576290986e-15 -29.0 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_29]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '0.8682408883346499 -4.924038765061041 0'
       end_point = '5.035797152340969 -28.559424837354037 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_30]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '1.7101007166283408 -4.698463103929543 0'
       end_point = '9.918584156444377 -27.251086002791347 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_31]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '2.5000000000000004 -4.330127018922193 0'
       end_point = '14.500000000000004 -25.11473670974872 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_32]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '3.2139380484326963 -3.8302222155948904 0'
       end_point = '18.640840680909637 -22.215288850450367 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_33]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '3.830222215594889 -3.213938048432698 0'
       end_point = '22.215288850450357 -18.640840680909648 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_34]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '4.330127018922192 -2.500000000000002 0'
       end_point = '25.114736709748712 -14.500000000000012 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_35]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '4.698463103929543 -1.710100716628343 0'
       end_point = '27.251086002791343 -9.918584156444389 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []

    [real_extensometer_v2_36]
       type = LineValueSampler
       warn_discontinuous_face_values = false
       start_point = '4.92403876506104 -0.8682408883346564 0'
       end_point = '28.55942483735403 -5.035797152341007 0'
       num_points = 7
       sort_by = x
       variable = 'disp_x disp_y stress_xx stress_xy stress_yy strain_xx strain_xy strain_yy'
       execute_on = TIMESTEP_END
    []
[]

[Positions]
  [all_elements]
    type = ElementCentroidPositions
  []
[]

[Outputs]
    exodus = true
    csv = true
[]

