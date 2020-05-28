import matplotlib.pyplot as plt

x = [0,0.5,1,1.5]
methods = ['GPR/SVR', 'TP-GMM', 'Original MDN', 'Entropy MDN']
fig, ax = plt.subplots()
width = 0.35
rects3 = ax.bar(x[2], 0.77, width, label='Original MDN')
rects4 = ax.bar(x[3], 0.81, width, label='Entropy MDN')
rects1 = ax.bar(x[0], 0.30, width, label='GPR/SVR')
rects2 = ax.bar(x[1], 0.45, width, label='TP-GMM')

ax.set_ylabel('Success Rate')
ax.set_xlabel('Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % (height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
fig.tight_layout()
plt.show()