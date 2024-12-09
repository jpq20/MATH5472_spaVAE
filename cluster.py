from utils import *


model_name = 'sample_151673'
data_file = f'/home/pjiangag/main/my_spaVAE/datasets/{model_name}.h5'
latent_file = f'/home/pjiangag/main/my_spaVAE/checkpoints/{model_name}_latent.txt'

model_name = latent_file.split('/')[-1].split('_')[1]

# Load data
final_latent = np.loadtxt(latent_file, delimiter=",")
data_mat = h5py.File(data_file, 'r')
pos = np.array(data_mat['pos']).astype('float64')
y = np.array(data_mat['Y']).astype('U26')  # ground-truth labels
data_mat.close()

# Filter out 'NA' labels
index = y != 'NA'

# Perform clustering
pred = KMeans(n_clusters=len(np.unique(y[index])), n_init=100).fit_predict(final_latent[index])
np.savetxt(f"/home/pjiangag/main/my_spaVAE/checkpoints/{model_name}_clustering_labels.txt", pred, delimiter=",", fmt="%i")

# Calculate NMI and ARI
nmi = np.round(metrics.normalized_mutual_info_score(y[index], pred), len(np.unique(y[index])))
ari = np.round(metrics.adjusted_rand_score(y[index], pred), len(np.unique(y[index])))
print("NMI:", nmi, "; ARI:", ari)

# Calculate distances
dis = pairwise_distances(pos[index], metric="euclidean", n_jobs=-1).astype(np.double)

# Refine clustering labels
pred_refined = refine(np.arange(pred.shape[0]), pred, dis, shape="hexagon")
np.savetxt(f"/home/pjiangag/main/my_spaVAE/checkpoints/{model_name}_refined_clustering_labels.txt", pred_refined, delimiter=",", fmt="%i")

# Calculate NMI and ARI for refined labels
nmi = np.round(metrics.normalized_mutual_info_score(y[index], pred_refined), len(np.unique(y[index])))
ari = np.round(metrics.adjusted_rand_score(y[index], pred_refined), len(np.unique(y[index])))
print("Refined NMI:", nmi, "; refined ARI:", ari)

# Prepare data for visualization
refined_pred = pred_refined + 1
pred_dat = {
    'refined_pred': refined_pred,
    'pos_x': pos[index, 0],
    'pos_y': pos[index, 1],
    'Y': y[index]
}

# Convert refined_pred to categorical type with specific levels
pred_dat['refined_pred'] = pd.Categorical(pred_dat['refined_pred'], categories=range(1, len(np.unique(y[index])) + 1))

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pred_dat, x='pos_x', y='pos_y', hue='refined_pred', palette='tab10', s=25, edgecolor=None)
plt.title('spaVAE predicted labels')
plt.axis('off')  # Hide axes
plt.legend(title='', loc='right', bbox_to_anchor=(1.15, 0.5), frameon=False, fontsize='large', markerscale=2)
plt.savefig(f"/home/pjiangag/main/my_spaVAE/checkpoints/{model_name}_latent_spots.png")
plt.show()

# Perform UMAP
umap_model = umap.UMAP(random_state=123)
umap_res = umap_model.fit_transform(final_latent[index])
spaVAE_umap = pd.DataFrame(umap_res, columns=['X1', 'X2'])
spaVAE_umap['Y'] = y[index]

# Plot the UMAP results
plt.figure(figsize=(10, 8))
sns.scatterplot(data=spaVAE_umap, x='X1', y='X2', hue='Y', legend='full')
plt.title("spaVAE Latent Embedding")
plt.xlabel("")
plt.ylabel("")
plt.axis('off')
plt.legend()
plt.savefig(f"/home/pjiangag/main/my_spaVAE/checkpoints/{model_name}_latent_embedding.png")
plt.show()
