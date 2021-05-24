import math
import numpy as np
import matplotlib.pyplot as plt
from verify import integral
from Vandermonde import do_c

means = [0, 1]   # 均值μ
sigmas = [math.sqrt(1), math.sqrt(4)]  # 标准差δ

for i in range(0, 2):
    for j in range(0, 2):
        x = np.linspace(means[i] - 3*sigmas[j], means[i] + 3*sigmas[j], 50)   # 定义域
        y = np.exp(-(x - means[i]) ** 2 / (2 * sigmas[j] ** 2)) / (math.sqrt(2*math.pi)*sigmas[j]) # 定义曲线函数
        plt.plot(x, y, "g", linewidth=2)    # 加载曲线
        print(integral(means[i], sigmas[j]))


# plt.grid(True)  # 网格线
# plt.show()  # 显示

van_c_8, ff_c_8, van_norm_8, ff_norm_8 = do_c(8)
van_c_16, ff_c_16, van_norm_16, ff_norm_16 = do_c(16)
print(van_c_8)
print(ff_c_8)
print(van_norm_8)
print(ff_norm_8)
print(van_c_16)
print(ff_c_16)
print(van_norm_16)
print(ff_norm_16)

