r0 = [1, 3, 5, 10, 20]
cne_auc = [0.6563205603255962, 0.6980907419549311, 0.7120363083514845, 0.7460940643045665, 0.7575413111838847]

cne_k_auc = [0.5920977685621336, 0.6467408284150646, 0.7032213947830543, 0.767792350664394, 0.820730268452297]
sine_auc = [0.4969899849065135, 0.5336982980302207, 0.5597678711742056, 0.6103424782741593, 0.6491826046395031]

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})


plt.plot(r0, cne_auc, marker='o', linestyle=":", linewidth=3, color="r", label=r"CNE")
plt.plot(r0, cne_k_auc, marker='h', linestyle=":", linewidth=3, color="g", label=r"CNE\_K")
plt.plot(r0, sine_auc, marker='p', linestyle=":", linewidth=3, color="b", label=r"SINE")
plt.xlabel(r"\% of Network Observed", fontsize=13)
plt.ylabel(r"Average ROC-AUC scores over multiple runs", fontsize=13)
plt.title(r"ROC-AUC scores vs \% Network Observed for different models", fontsize=14)
plt.legend()

plt.savefig("expt1.png", dpi=600)
plt.show()