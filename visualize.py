import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config


if __name__ == '__main__':
    statistics_data = pd.read_json(config.statistics_path)
    plt.figure(figsize=(16, 6))
    plt.title('Accuracy and Loss During the Train')
    plt.xlabel('Epoch')
    sns.lineplot(data=statistics_data)
    plt.show()
