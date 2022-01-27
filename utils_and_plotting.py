# Importing libreries

from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import NeighborhoodComponentsAnalysis 
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance


def quick_cleaning(recon):
    recon = recon > threshold_otsu(recon)
    recon = morphology.remove_small_objects(recon, min_size=100)
    return recon

def LoadPots(path):
    pots = sparse.load_npz(path).toarray().reshape(-1, 256, 256)
    return pots


def PlotRecostruction(outputs, start = 0, stop = 50, plt_range = 10, plt_number = 10, Training = True, QuickCleaning = False, SaveFig = False, fontsize = 10):
    for k in range(start, stop, plt_range):


        imgs = outputs[k][1].to("cpu").detach().numpy()
        recon = outputs[k][2].to("cpu").detach().numpy()


        row_number = 3 if QuickCleaning else 2
        fig, ax = plt.subplots(row_number, int(plt_number), figsize=(15, 5),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.05, wspace=0.1))
        plot_title = (f"Epoch {k+1}") if Training else ("Test")
        plt.suptitle(plot_title)
        if QuickCleaning:
            for i in range(plt_number):
                ax[0, i].imshow(np.transpose(imgs[i], (1,2,0)), cmap='binary_r')
                ax[1, i].imshow(np.transpose(recon[i],(1,2,0)), cmap='binary_r')
                ax[2, i].imshow(quick_cleaning(np.transpose(recon[i],(1,2,0))), cmap='binary_r')
                ax[0, 0].set_ylabel('Original')
                ax[1, 0].set_ylabel('Reconstructed')
                ax[2, 0].set_ylabel('Reconstructed and \n cleaned');
                ax[2, i].set_xlabel(i, fontsize = fontsize)
        else :    
            for i in range(int(plt_number)):
                ax[0, i].imshow(np.transpose(imgs[i], (1,2,0)), cmap='binary_r')
                ax[1, i].imshow(np.transpose(recon[i],(1,2,0)), cmap='binary_r')
                ax[0, 0].set_ylabel('Original')
                ax[1, 0].set_ylabel('Reconstructed');
                ax[1, i].set_xlabel(i, fontsize = fontsize)

        if SaveFig:
            fig.savefig(f"Rec_{k}.jpg", dpi = 300)


def PrincipalComponentAnalysis(data, n_components=2):
    pca = PCA(n_components= n_components).fit(data)
    reduced = pca.fit_transform(data)
    return reduced

def Umap(data, n_components=2):
    ump = umap.UMAP(n_components= n_components).fit(data)
    embedded = ump.fit_transform(data)
    return embedded

def NCA(data, y, n_components = 2):
    nca = NeighborhoodComponentsAnalysis(n_components, init = "pca", random_state=1).fit(data, y)
    reduction = nca.fit_transform(data, y)
    return reduction

def PlottingEmbeddings(data,archeo_info, labels = "Functional_class", SaveFig=False):
    reduced = PrincipalComponentAnalysis(data)
    embedded = Umap(data)
    fig, ax = plt.subplots(1, 2, figsize = (20,10))
    sns.scatterplot(data = archeo_info, x =embedded[:,0], y = embedded[:,1], s = 5, hue =archeo_info[labels], ax = ax[1])
    ax[1].set_title("UMAP")
    ax[1].set_ylabel('Embedding 2')
    ax[1].set_xlabel('Embedding 1')
    sns.scatterplot(data = archeo_info, x =reduced[:,0], y = reduced[:,1], s = 5, hue =archeo_info[labels], ax = ax[0])
    ax[0].set_title("PCA")
    ax[0].set_ylabel('PC 2')
    ax[0].set_xlabel('PC 1')

    if SaveFig:
        fig.savefig(f"Reduction.jpg", dpi = 300)

def KdePlot(data, archeo_info, subsampling = True, SaveFig=False):
    chronology = ["FBA", "EIA1", "EIA2", "OP"]
    hue_order = ["Etruria", "Latium"]

    with pd.option_context('mode.chained_assignment',None):
        fig, axs = plt.subplots(1, len(chronology), sharex="row", sharey="row", figsize = (20,5))

        for x, i in enumerate(chronology):
            info_selected_chrono = archeo_info[(archeo_info.chronology == i)]
            pots_chrono = data.loc[info_selected_chrono.index]

            if subsampling == True:

                min_number = min(info_selected_chrono.value_counts("Region"))
                max_index = np.argmax(info_selected_chrono.value_counts("Region"))
                max_region = info_selected_chrono.value_counts("Region").index[max_index]

                info_selected_chrono_max_region = info_selected_chrono[(info_selected_chrono.Region == max_region)]
                info_selected_chrono_min_region = info_selected_chrono[(info_selected_chrono.Region != max_region)]
                random_select = info_selected_chrono_max_region.sample(n = min_number)
                
                list1, list2 = list(info_selected_chrono_min_region.index), list(random_select.index)

                list1.extend(list2)

                info_selected_chrono = info_selected_chrono.loc[list1]
                pots_chrono = pots_chrono.loc[list1]

            reduction = NCA(pots_chrono, info_selected_chrono.Region, 1)
            lda_comp_norm = MinMaxScaler().fit_transform(reduction)
            df_values = pd.DataFrame(lda_comp_norm, columns=[f"Dim_{dim}" for dim in range(reduction.shape[1])] , index = info_selected_chrono.index)
                

            info_selected_chrono_joined = info_selected_chrono.join(df_values)
            
            sns.kdeplot(data = info_selected_chrono_joined, x = "Dim_0", hue = info_selected_chrono_joined.Region, legend = True, ax = axs[x], hue_order = hue_order)
            axs[x].set_title(i)

    if SaveFig:
        fig.savefig(f"Kde_plot.jpg", dpi = 300)




def WSDist(data, archeo_info, resampling_number = 50, PlotType = "violin", show_points = False, SaveFig = False):
    pipeline = []
    distance_ws = []
    
    chronology = ["FBA", "EIA1", "EIA2", "OP"]
    for n in range(resampling_number):
        with pd.option_context('mode.chained_assignment',None):

            for _, i in enumerate(chronology):
                info_selected_chrono = archeo_info[(archeo_info.chronology == i)]
                pots_chrono = data.loc[info_selected_chrono.index]

                min_number = min(info_selected_chrono.value_counts("Region"))
                max_index = np.argmax(info_selected_chrono.value_counts("Region"))
                max_region = info_selected_chrono.value_counts("Region").index[max_index]

                info_selected_chrono_max_region = info_selected_chrono[(info_selected_chrono.Region == max_region)]
                info_selected_chrono_min_region = info_selected_chrono[(info_selected_chrono.Region != max_region)]
                random_select = info_selected_chrono_max_region.sample(n = min_number)
                    
                list1, list2 = list(info_selected_chrono_min_region.index), list(random_select.index)

                list1.extend(list2)

                info_selected_chrono = info_selected_chrono.loc[list1]
                pots_chrono = pots_chrono.loc[list1]                   

                reduction = NCA(pots_chrono, info_selected_chrono.Region, 1)
                nca_norm = MinMaxScaler().fit_transform(reduction)
                df_values = pd.DataFrame(nca_norm, columns=[f"Dim_{dim}" for dim in range(reduction.shape[1])] , index = info_selected_chrono.index)
                    

                info_selected_chrono_joined = info_selected_chrono.join(df_values)

                

                pipeline.append(info_selected_chrono_joined)



            total_df = pd.concat(pipeline)



            for _, i in enumerate(chronology):
                total_df_chr = total_df[(total_df.chronology == i)]
                reg = total_df_chr[["Region", "Dim_0"]]
                reg_lat = reg[reg.Region == "Latium"]
                reg_etr = reg[reg.Region == "Etruria"]
                w_d = wasserstein_distance(reg_lat["Dim_0"], reg_etr["Dim_0"])



                distance_ws.append((n, i, w_d))

    df_dist = pd.DataFrame(distance_ws, columns =  [["Run", "Epochs", "Wasserstein"]])
    
    fig, ax = plt.subplots(1, 1, figsize = (10,5))
    if PlotType == "boxplot":        
        ax = sns.boxplot(x = df_dist["Epochs"].values.reshape(-1), y =df_dist["Wasserstein"].values.reshape(-1), showmeans=True,meanprops={"marker":"o",
                            "markerfacecolor":"black", 
                            "markeredgecolor":"black",
                            "markersize":"10"})
        if show_points:
            ax = sns.stripplot(x=df_dist["Epochs"].values.reshape(-1), y=df_dist["Wasserstein"].values.reshape(-1),  linewidth=1)
        
        
    elif PlotType == "violin":
        ax = sns.violinplot(x=df_dist["Epochs"].values.reshape(-1), y=df_dist["Wasserstein"].values.reshape(-1))
        if show_points:
            ax = sns.stripplot(x=df_dist["Epochs"].values.reshape(-1), y=df_dist["Wasserstein"].values.reshape(-1),  linewidth=1)
    
    elif PlotType == "points":
        ax = sns.stripplot(x=df_dist["Epochs"].values.reshape(-1), y=df_dist["Wasserstein"].values.reshape(-1),  linewidth=1)
    
    else: assert "You can use 'violin', 'boxplot' or 'points' as graph type"
    
    

    if SaveFig:
        fig.savefig(f"WS_plot.jpg", dpi = 300)




    