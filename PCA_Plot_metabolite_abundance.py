import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
import matplotlib.transforms as transforms


def confidence_ellipse(
    x,
    y,
    ax,
    n_std = 3.0,
    facecolor = 'none',
    **kwargs
    ):
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width = ell_radius_x * 2,
            height = ell_radius_y * 2,
            facecolor = facecolor,
            **kwargs
            )

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
        # render plot with "plt.show()".



#Loading the data
met_data = pd.read_csv('Metabolite_abundance_matrix.csv')

sort_list = [
    'DMSO 1-3 Basal_1',
    'DMSO 1-3 Basal_2',
    'DMSO 1-3 Basal_3',
    'DMSO 3-5 Basal_1',	
    'DMSO 3-5 Basal_2',	
    'DMSO 3-5 Basal_3',	
    'DMSO 5-7 Basal_1',	
    'DMSO 5-7 Basal_2',
    'DMSO 5-7 Basal_3'
    ]

pc_labels = [
    '1-3 Days',
    '1-3 Days',
    '1-3 Days',
    '3-5 Days',
    '3-5 Days',
    '3-5 Days',
    '5-7 Days',
    '5-7 Days',
    '5-7 Days'
    ]

pc_colors = [
    'grey',
    'grey',
    'grey',
    'b',
    'b',
    'b',
    'r',
    'r',
    'r'
    ]

#Running PCA on Metabolite abundance without luminal cells

X_data_pre = met_data[sort_list].values
X_data = np.transpose(X_data_pre)
X_data_unnormalized = StandardScaler()
X_data = X_data_unnormalized.fit_transform(X_data)
X_data[np.isnan(X_data)] = 0

#Running PCA
components = PCA(n_components = 2)
components.fit(X_data)
X = components.transform(X_data)

PC1_values = []
PC2_values = []
for i in range(0, len(X)):
    PC1_values.append(X[i][0])
    PC2_values.append(X[i][1])
    
    
#Constructing the PCA Plot
fig = plt.figure()
ax = fig.add_subplot(
    1,
    1, 
    1
    )
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['right'].set_linestyle((
    0,
    (4, 4)
    ))
ax.spines['top'].set_linestyle((
    0,
    (4, 4)
    ))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_position('center')
ax.spines['top'].set_position('center')
# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
confidence_ellipse(
    np.array(PC1_values[:3]),
    np.array(PC2_values[:3]),
    ax,
    n_std = 2.0,
    facecolor = 'grey',
    alpha = 0.25,
    edgecolor = 'grey'
    )
confidence_ellipse(
    np.array(PC1_values[:3]),
    np.array(PC2_values[:3]),
    ax,
    n_std = 2.0,
    facecolor = 'None',
    edgecolor = 'k'
    )
confidence_ellipse(
    np.array(PC1_values[3:6]),
    np.array(PC2_values[3:6]),
    ax,
    n_std = 2.0,
    facecolor = 'b',
    alpha = 0.25,
    edgecolor = 'b'
    )
confidence_ellipse(
    np.array(PC1_values[3:6]),
    np.array(PC2_values[3:6]),
    ax,
    n_std = 2.0,
    facecolor = 'None',
    edgecolor = 'k'
    )
confidence_ellipse(
    np.array(PC1_values[6:]),
    np.array(PC2_values[6:]),
    ax,
    n_std = 2.0,
    facecolor = 'r',
    alpha = 0.25,
    edgecolor = 'r'
    )
confidence_ellipse(
    np.array(PC1_values[6:]),
    np.array(PC2_values[6:]),
    ax,
    n_std = 2.0,
    facecolor = 'None',
    edgecolor = 'k'
    )
plt.scatter(
    PC1_values,
    PC2_values,
    label = pc_labels,
    marker = '+',
    color = pc_colors
    )
plt.xlabel('PC1 (' + str(round(components.explained_variance_ratio_[0] * 100, 2)) + '%)')
plt.ylabel('PC2 (' + str(round(components.explained_variance_ratio_[1] * 100, 2)) + '%)')
legend_elements = [
    Patch(
        facecolor = 'grey',
        edgecolor = 'k',
        label = '1-3 Days'
        ),
    Patch(
        facecolor = 'b',
        edgecolor = 'k',
        label = '3-5 Days'
        ),
    Patch(
        facecolor = 'r',
        edgecolor = 'k',
        label = '5-7 Days'
        )
    ]
plt.legend(handles = legend_elements)
plt.show()