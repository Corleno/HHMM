import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("model_validation_version0", "rb") as res:
	inv_x_observed, inv_y_observed, x_grids, y_original_mat, y_mixed_mat = pickle.load(res)

y_mixed_quantiles = np.percentile(y_mixed_mat, [2.5, 50, 97.5], axis=0)
y_original_quantiles = np.percentile(y_original_mat, [2.5, 50, 97.5], axis=0)


plt.figure()
plt.ylim(0.6, 1)
plt.xlim(0, 15)
plt.plot(inv_x_observed, inv_y_observed, color = 'k')
# plot original curves
plt.plot(x_grids, y_original_quantiles[1,:], color = 'b')
plt.plot(x_grids[:,np.newaxis], y_original_quantiles[[0,2],:].T, color = 'b', linestyle='--')
# plot mixed curves
plt.plot(x_grids, y_mixed_quantiles[1,:], color = 'r')
plt.plot(x_grids[:,np.newaxis], y_mixed_quantiles[[0,2],:].T, color = 'r', linestyle='--')
# plt.legend(["Obeserved KM curve", "KM curve from the simple HMM", "KM curve from the hierarchical HMM"], loc=3, fontsize = 'x-small')        
plt.ylabel('Probability')
plt.xlabel('Time [years]')
# plt.show()
plt.savefig("KM_curve_shared_version0.png")


with open("model_validation_version1", "rb") as res:
	inv_x_observed, inv_y_observed, x_grids, y_original_mat, y_mixed_mat = pickle.load(res)

y_mixed_quantiles = np.percentile(y_mixed_mat, [2.5, 50, 97.5], axis=0)
y_original_quantiles = np.percentile(y_original_mat, [2.5, 50, 97.5], axis=0)


plt.figure()
plt.ylim(0.8, 1)
plt.xlim(0, 15)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=12)


plt.plot(inv_x_observed, inv_y_observed, color = 'k', linewidth = 2.0)
# plot original curves
plt.plot(x_grids, y_original_quantiles[1,:], color = 'b', linewidth = 2.0)
plt.plot(x_grids[:,np.newaxis], y_original_quantiles[[0,2],:].T, color = 'b', linestyle='--', linewidth = 2.0)
# plot mixed curves
plt.plot(x_grids, y_mixed_quantiles[1,:], color = 'r', linewidth = 2.0)
plt.plot(x_grids[:,np.newaxis], y_mixed_quantiles[[0,2],:].T, color = 'r', linestyle='--', linewidth = 2.0)
# plt.legend(["Obeserved KM curve", "KM curve from the simple HMM", "KM curve from the hierarchical HMM"], loc=3, fontsize = 'x-small')        
plt.ylabel('Probability', fontsize = 20)
plt.xlabel('Time [years]', fontsize = 20)

# plt.show()
plt.savefig("KM_curve_shared_version1.png")
