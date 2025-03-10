import csdl_alpha as csdl
import jax
import numpy as np
import csdml
import optax
import pickle

recorder = csdl.Recorder(inline=True)
recorder.start()

def import_training_data(filename):
    training_data_inputs_1 = []
    training_data_inputs_2 = []
    training_data_inputs_3 = []
    training_data_outputs_1 = []
    training_data_outputs_2 = []
    import h5py
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        # data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        for sample_name in f.keys():
            sample_group = f[sample_name]
            sample_width_shape_variables = sample_group['width_shape_deltas'][()].reshape((-1,1))
            training_data_inputs_1.append(sample_width_shape_variables)
            sample_pump_pressure = sample_group['base_max_pressure'][()].reshape((-1,1))
            training_data_inputs_2.append(sample_pump_pressure)
            sample_displacement = sample_group['variable_0'][()].reshape((-1,1))
            training_data_outputs_1.append(sample_displacement)
            sample_applied_work = sample_group['applied_work'][()].reshape((-1,1))
            training_data_outputs_2.append(sample_applied_work)

            if filename == 'saved_data_6_dims.hdf5':
                sample_height = sample_group['height'][()].reshape((-1,1))
                training_data_inputs_3.append(sample_height)
        training_data_inputs_1 = np.hstack(training_data_inputs_1)
        training_data_inputs_2 = np.hstack(training_data_inputs_2)
        training_data_outputs_1 = np.hstack(training_data_outputs_1)
        training_data_outputs_2 = np.hstack(training_data_outputs_2)
        return training_data_inputs_1, training_data_inputs_2, training_data_inputs_3, \
            training_data_outputs_1, training_data_outputs_2

# region train ML model

# region load training/test data
training_data_width_shape_variables_5_dims, training_data_pump_pressure_5_dims,_, \
    training_data_displacements_5_dims, training_data_applied_work_5_dims = \
    import_training_data('examples/advanced_examples/robotic_fish/training_data/saved_data_5_dims.hdf5')
training_data_width_shape_variables_6_dims, training_data_pump_pressure_6_dims, training_data_height_6_dims, \
    training_data_displacements_6_dims, training_data_applied_work_6_dims = \
    import_training_data('examples/advanced_examples/robotic_fish/training_data/saved_data_6_dims.hdf5')
training_data_width_shape_variables = np.hstack([training_data_width_shape_variables_5_dims, training_data_width_shape_variables_6_dims])
training_data_pump_pressure = np.hstack([training_data_pump_pressure_5_dims, training_data_pump_pressure_6_dims])
training_data_displacements = np.hstack([training_data_displacements_5_dims, training_data_displacements_6_dims])
training_data_applied_work = np.hstack([training_data_applied_work_5_dims, training_data_applied_work_6_dims])

# # region oops
# # I realize, I forgot to add the direct model input as an output to the generator, so I need to map
# # the geometry inputs to the structural solver inputs
# additional_inputs = [width_shape_variables, height]
# additional_outputs = [structural_mesh_displacements]

# sim = csdl.experimental.JaxSimulator(
#     recorder = recorder,
#     additional_inputs=additional_inputs,
#     additional_outputs=additional_outputs,
#     gpu=False
# )

# training_data_mesh_displacements = []
# for i in range(training_data_width_shape_variables.shape[1]):
#     print('Simulating sample: ', i, ' of ', training_data_width_shape_variables.shape[1])
#     sim[width_shape_variables] = training_data_width_shape_variables[:,i]
#     if i > training_data_width_shape_variables_5_dims.shape[1]:
#         sim[height] = training_data_height_6_dims[:,i-training_data_width_shape_variables_5_dims.shape[0]]
#     sim.run()
#     training_data_mesh_displacements.append(sim[structural_mesh_displacements].reshape((-1,1)))
# training_data_mesh_displacements = np.hstack(training_data_mesh_displacements)
# # Save correct mesh displacements
# np.save('training_data_mesh_displacements.npy', training_data_mesh_displacements)
# exit()
# # endregion oops

training_data_geometry_coefficients = np.load('examples/advanced_examples/robotic_fish/training_data/training_data_geometry_coefficients.npy')
# training_data_mesh_displacements = np.load('examples/advanced_examples/robotic_fish/training_data/training_data_mesh_displacements.npy')
num_test_data = 50
test_data_geometry_coefficients = training_data_geometry_coefficients[:, -num_test_data:]
# test_data_mesh_displacements = training_data_mesh_displacements[:, -num_test_data:]
test_data_pump_pressure = training_data_pump_pressure[:, -num_test_data:]
test_data_displacements = training_data_displacements[:, -num_test_data:]
test_data_applied_work = training_data_applied_work[:, -num_test_data:]

training_data_geometry_coefficients = training_data_geometry_coefficients[:, :-num_test_data]
# training_data_mesh_displacements = training_data_mesh_displacements[:, :-num_test_data]
training_data_pump_pressure = training_data_pump_pressure[:, :-num_test_data]
training_data_displacements = training_data_displacements[:, :-num_test_data]
training_data_applied_work = training_data_applied_work[:, :-num_test_data]

training_data_inputs = np.vstack([training_data_geometry_coefficients, training_data_pump_pressure]).T
test_data_inputs = np.vstack([test_data_geometry_coefficients, test_data_pump_pressure]).T
# training_data_inputs = np.vstack([training_data_mesh_displacements, training_data_pump_pressure]).T
# test_data_inputs = np.vstack([test_data_mesh_displacements, test_data_pump_pressure]).T
training_data_outputs = np.vstack([training_data_displacements, training_data_applied_work]).T
test_data_outputs = np.vstack([test_data_displacements, test_data_applied_work]).T

loss_data = (training_data_inputs, training_data_outputs)
test_data = (test_data_inputs, test_data_outputs)
# endregion load training/test data

# device = jax.devices('gpu')[0]
device = jax.devices('cpu')[0]
input_size = training_data_geometry_coefficients.shape[0] + 1 # +1 for pump_pressure
# input_size = training_data_mesh_displacements.shape[0] + 1 # +1 for pump_pressure
output_size = training_data_displacements.shape[0] + 1 # +1 for applied_work
# output_size = structural_displacements.size # start with just displacements for now
# model = csdml.FCNN(input_dim=output_size, hidden_dims=[200, 200, 200], output_dim=output_size, activation=['relu', 'relu', 'relu', None])
model = csdml.FCNN(input_dim=input_size, hidden_dims=[320, 320, 320], output_dim=output_size, activation=['relu', 'relu', 'relu', None])

optimizer = optax.adam(1e-3)
loss_history, test_loss_history, best_param_vals = model.train_jax_opt(optimizer, loss_data, test_data=test_data, 
                                                                       device=device, num_epochs=400, num_batches=1, plot=False)
# save loss history
loss_history = np.array(loss_history)
test_loss_history = np.array(test_loss_history)

np.savez('loss_history.npz', loss_history=loss_history, test_loss_history=test_loss_history)
pickle.dump(best_param_vals, open('best_param_vals.pkl', 'wb'))

# plot the loss history
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
__=ax.plot(np.log10(loss_history))
if test_data is not None:
    __=ax.plot(np.log10(test_loss_history))
    ax.legend(['train', 'test'])
xlabel = ax.set_xlabel(r'${\rm step\ number}$')
ylabel = ax.set_ylabel(r'$\log_{10}{\rm loss}$')
title = ax.set_title(r'${\rm training\ history}$')
plt.savefig('loss_plot.png')
plt.show()