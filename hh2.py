
wc = WordCloud()
wc.generate(''.join(data_cutted['Comment'][data_cutted['Class'] == -1]))
plt.axis('off')
plt.imshow(wc)
plt.show()