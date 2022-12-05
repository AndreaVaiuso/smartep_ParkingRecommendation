import matplotlib.pyplot as plt

# INPUT:

out_plots_dirpath = "out/plots"

results = {"pfo": {"accuracy": [0.2693298969072165, 0.2757731958762887, 0.28350515463917525, 0.29381443298969073], "hit_rate": [0.2693298969072165, 0.2693298969072165, 0.2693298969072165, 0.2693298969072165], "precision": [0.7925257731958762, 0.4097938144329897, 0.27849369988545436, 0.21176975945017185], "recall": [0.210706103747341, 0.21863136147929962, 0.22591228931435112, 0.23611417934871548], "satisfaction": [0.45203593844227347, 0.45203593844227347, 0.45595531584983484, 0.46661561759967296], "f1-score": [0.33290413039557665, 0.28513746109937776, 0.24946234044744803, 0.22328035739591218]}, "p": {"accuracy": [0.9716129032258064, 0.9987096774193548, 1.0, 1.0], "hit_rate": [0.9716129032258064, 0.9716129032258064, 0.9716129032258064, 0.9716129032258064], "precision": [0.9883870967741936, 0.5293548387096774, 0.35985663082437436, 0.28129032258064535], "recall": [0.9245161290322581, 0.9651612903225807, 0.9696774193548388, 0.9819354838709677], "satisfaction": [0.0, 0.0, 0.0, 0.0], "f1-score": [0.9553853016373823, 0.6837166749058297, 0.5249129934934282, 0.43730732478829637]}, "fo": {"accuracy": [0.17396907216494845, 0.327319587628866, 0.4690721649484536, 0.5451030927835051], "hit_rate": [0.17396907216494845, 0.17396907216494845, 0.17396907216494845, 0.17396907216494845], "precision": [0.6430412371134021, 0.5905283505154639, 0.5391609392898045, 0.4725891323024058], "recall": [0.1602550728195059, 0.30309176075928695, 0.43640310505645546, 0.5203690067092129], "satisfaction": [0.3613405312986267, 0.5524256874774652, 0.6680446758481995, 0.747275072226572], "f1-score": [0.2565693854317849, 0.4005824740911167, 0.4823701926999537, 0.49532951631286076]}}

if out_plots_dirpath[-1] != "/":
    out_plots_dirpath += "/"

map_keys_labels = {
    "accuracy": "Accuracy",
    "hit_rate": "Hit Rate",
    "precision": "Precision",
    "recall": "Recall",
    "satisfaction": "Satisfaction",
    "f1-score": "F1-Score",
    "coverage": "Coverage"
}

# for each metric, one bar for pfo, one for p and one for fo
for metric in results["pfo"]:
    title = map_keys_labels[metric] if metric not in ["precision", "recall"] else "Mean Average " + map_keys_labels[metric]
    plt.figure()
    plt.title(title)
    plt.xlabel("Use case")
    plt.ylabel(map_keys_labels[metric])
    plt.xticks(range(3), ["PFO", "P", "FO"])

    plt.ylim(0, 1)

    plt.bar(0, results["pfo"][metric][-1], color="blue", label="PFO")
    plt.bar(1, results["p"][metric][-1], color="red", label="P")
    plt.bar(2, results["fo"][metric][-1], color="green", label="FO")
    #plt.legend()
    plt.savefig(out_plots_dirpath + title.lower().replace(" ", "_") + ".pdf")
    plt.close()


# precision, satisfaction, f1-score when the index varies
for metric in ["precision", "satisfaction", "f1-score"]:
    plt.figure()
    title = map_keys_labels[metric] + "@k"
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(map_keys_labels[metric])

    plt.xticks([ x + 0.2 for x in range(1, len(results["pfo"][metric]) + 1)], range(1, len(results["pfo"][metric]) + 1))
    
    for i, z in enumerate(["pfo", "p", "fo"]):
        xs = [1 + 0.2 * i + 1 * x for x in range(len(results[z][metric]))]
        plt.bar(xs, [results[z][metric][k] for k in range(len(results[z][metric]))], width=0.2, label=z.upper())

    plt.ylim(0, 1)

    plt.legend()
    plt.savefig(out_plots_dirpath + metric + "_at_k" + ".pdf")
    plt.close()