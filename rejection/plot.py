import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

def plot_norm_distribution():
    # 设置均值和标准差
    mean = 512
    std_dev = 12

    # 创建一个包含1000个点的x值范围，从400到600
    x = np.linspace(400, 600, 1000)

    # 使用正态分布公式计算y值
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    # 绘制正态分布曲线
    plt.plot(x, y)
    plt.axvline(x=512, color='g', linestyle='-', ymax=0.11) 
    plt.text(mean, 0.037, 'model_max_input', color='g', ha='center', va='bottom')
    plt.ylim(0, 0.3)
    plt.xlim(0, 600)
    # 设置图表的标题和标签
    plt.xlabel('token_length')
    plt.ylabel('freqency')

    # 显示图表
    plt.show()

def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for jsonl_file in os.listdir('./'):

        
        if not jsonl_file.endswith('.jsonl'):
            continue

        if not 'chunk_size' in jsonl_file:
            continue

        x = []
        y = []
        z = []
        print(jsonl_file)

        datas = []
        with open(jsonl_file) as f:
            for json_str in f:
                json_obj = json.loads(json_str)
                datas.append(json_obj)
        
        datas.sort(key=lambda x: x['throttle'])

        for data in datas:
                chunk_size = data['chunk_size']
                throttle = data['throttle']
                f1 = data['f1']

                x.append(chunk_size)
                y.append(throttle)
                z.append(f1)

        # 绘制3D曲线
        ax.plot(x, y, z)

    # 添加标题和标签
    ax.set_title('3D Line Plot')
    ax.set_xlabel('chunk_size')
    ax.set_ylabel('throttle')
    ax.set_zlabel('f1')

    # 显示图形
    plt.show()


def plot_cross_splitter():
    fig = plt.figure()
    for splitter in os.listdir('./'):

        if not splitter.startswith('chunk_size'):
            continue

        if not os.path.isdir(splitter):
            continue

        items = []
        for jsonl_file in os.listdir(splitter):
            if not 'chunk_size' in jsonl_file:
                continue

            print(splitter, jsonl_file)
            datas = []

            with open(os.path.join(splitter, jsonl_file)) as f:
                for json_str in f:
                    json_obj = json.loads(json_str)
                    datas.append(json_obj)
            datas.sort(key=lambda x: x['f1'])

            items.append({
                'chunk_size': datas[-1]['chunk_size'],
                'f1': datas[-1]['f1']
            })

        items.sort(key=lambda x: x['chunk_size'])
        x = []
        y = []
        for item in items:
            if item['chunk_size'] > 1000:
                continue
            x.append(item['chunk_size'])
            y.append(item['f1'])
        print(x, y)
        # 绘制曲线
        label_name = splitter.split('chunk_size_')[-1]
        plt.plot(x, y, label=label_name)

    # 添加标题和标签
    plt.xlabel('chunk_size')
    plt.ylabel('best_f1')
    plt.legend()
    # 显示图形
    plt.show()

def plot_chinese_splitter_hist():

    import matplotlib.pyplot as plt

    # 定义区间的边界和对应的百分比
    bins = [0, 898, 1796, 2694, 3592, 4490, 5388, 6286, 7184, 8082, 8980]
    percentages = [99.97, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01]

    # 绘制直方图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.bar(range(len(bins) - 1), percentages, width=1, edgecolor='black')

    # 设置图形的标题和坐标轴标签
    plt.title('Chinese Splitter Distribution')
    plt.xlabel('Range')
    plt.ylabel('Percentage')

    # 显示区间标签
    plt.xticks(range(len(bins) - 1), [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)])

    # 显示每个条形的百分比
    for i in range(len(percentages)):
        plt.text(i, percentages[i] + 0.5, str(percentages[i]))

    # 显示图形
    plt.show()

def plot_recursive_text_splitter_hist():
    import matplotlib.pyplot as plt

    # 定义区间的边界和对应的百分比
    bins = [0, 76, 152, 228, 304, 380, 456, 532, 608, 684, 760, 836]
    percentages = [1.83, 2.35, 1.94, 1.75, 2.43, 4.65, 6.06, 6.88, 13.43, 46.19, 12.50]

    # 绘制直方图
    plt.figure(figsize=(12, 6))  # 设置图形大小
    plt.bar(range(len(bins) - 1), percentages, width=1, edgecolor='black')

    # 设置图形的标题和坐标轴标签
    plt.title('Recursive Text Splitter Distribution')
    plt.xlabel('Range')
    plt.ylabel('Percentage')

    # 显示区间标签
    plt.xticks(range(len(bins) - 1), [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)])

    # 显示每个条形的百分比
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage + 0.5, f'{percentage:.2f}%', ha='center')

    # 优化y轴的刻度，使其更加清晰
    plt.ylim(0, max(percentages) + 5)

    # 显示图形
    plt.show()

def classification():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import LogisticRegression

    # 生成更均匀分布的数据点
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=2.0, random_state=42)

    # 将数据分为红色点和蓝色点
    red_x, red_y = X[y == 0], y[y == 0]
    blue_x, blue_y = X[y == 1], y[y == 1]

    # 绘制红色点
    plt.scatter(red_x[:, 0], red_x[:, 1], c='green', label='Positive')

    # 绘制蓝色点
    plt.scatter(blue_x[:, 0], blue_x[:, 1], c='pink', label='Negative')

    # 使用逻辑回归来找到决策边界
    model = LogisticRegression()
    model.fit(X, y)

    # 绘制决策边界
    x_vals = np.linspace(-10, 10, 100).reshape(100, 1)
    y_vals = model.coef_[0][0] * x_vals + model.intercept_[0]
    plt.plot(x_vals, y_vals, c='black', label='Throttle')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()

if __name__ == '__main__':
    classification()
