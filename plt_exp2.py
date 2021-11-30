strats = ['random\_1', 
        'random\_2',  
        'random\_3', 
        'pagerank',  
        'max\_degree\_sum', 
        'max\_probability', 
        'min\_distance',
        'max\_entropy',]

#cne_k
wo_distwt = [0.8602498661355961, 0.8522626298968556, 0.8589788448397591, 0.8507833548547058, 0.8601833535546239, 0.8674077625473229, 0.8762984331951516, 0.8752096830026682]
w_distwt = [0.8576071133423423, 0.8494301573422476, 0.8577445915436665, 0.8485491231244785, 0.8590612322352613, 0.8695400050093345, 0.8897307575867796, 0.8753328064675422]

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})

r0 = range(8)
plt.figure(figsize=[10, 6])
plt.plot(r0, wo_distwt, marker='o', linestyle=":", linewidth=3, color="r", label=r"Without Distance Weighting")
plt.plot(r0, w_distwt, marker='h', linestyle=":", linewidth=3, color="g", label=r"With Distance Weighting")
plt.xticks(list(r0), strats)
plt.xlabel(r"Query Strategies", fontsize=13)
plt.ylabel(r"Average ROC-AUC scores", fontsize=13)
plt.title(r"Average ROC-AUC scores vs Different Query Strategies", fontsize=14)
plt.legend()
plt.savefig("expt2.png", dpi=600)
plt.show()