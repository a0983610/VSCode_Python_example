import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

"""
 1.參數設定
"""
L, sep, N, k, cut = 6, 2, 500, 5, 5 # 0.5倍寬度, 波源與原點距離, 每邊分割數量, 角波數, z軸範圍
fps, frn = 24, 50           # 每秒影格數量, 影格總數
x = np.linspace(-L, L, N)   # x軸
y = np.linspace(-L, L, N)   # y軸
X, Y = np.meshgrid(x, y)    # 組成2維陣列
j = np.complex(0, 1)        # 根號 -1

"""
 2.設定計算振幅的函數, 計算每個位置的振幅並存入陣列
"""
# 自訂函式, 計算每個位置對應的振幅
def func(x, y, t):
    r1 = np.sqrt((x-sep)**2 + y**2)    # 點波源1
    r2 = np.sqrt((x+sep)**2 + y**2)    # 點波源2
    z = np.exp(j*k*r1)/r1 + np.exp(j*k*r2)/r2    # 兩個點波源的 Green's function 總合
    return np.real(z*np.exp(-j*t))    # 回傳實部

Z = np.zeros((N, N, frn))   # 儲存振幅用的2維陣列
T = np.linspace(0, 2*np.pi, frn)   # 儲存時間用的1維陣列

# 計算每個時刻每個位置對應的振幅, 有加cut效果較佳
for i in range(frn):
    Z[:, :, i] = func(X, Y, T[i]).clip(-cut, cut)

"""
 3.繪圖
"""
fig = plt.figure(figsize=(8, 6), dpi=100)   # 開啟繪圖視窗
ax = fig.gca(projection='3d')   # 設定為 3D 繪圖
ax.set_zlim(-cut, cut)   # 設定z軸範圍
ax.zaxis.set_major_locator(LinearLocator(10))   # 將z軸分為10格
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))   # 設定z軸數值格式

# 以某個預設的colormap為基底, 修改成對應到 -cut ~ +cut 的colormap
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(np.arange(-cut, cut, 0.1))

# 自訂函式, 先移除前一張圖, 再畫出下一張圖
def update(frame_number):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, Z[:, :, frame_number], cmap=mappable.cmap,
                              norm=mappable.norm, linewidth=0.5, antialiased=False, alpha=0.7)

# t = 0 的圖片
plot = [ax.plot_surface(X, Y, Z[:, :, 0], cmap=mappable.cmap, norm=mappable.norm,
                        linewidth=0.5, antialiased=False, alpha=0.7)]

# 顯示數值及色階對應方式
fig.colorbar(mappable, shrink=0.5, aspect=5)

# 產生動畫, 目標為繪圖物件fig, 使用自訂函式update更新圖片, 圖片總數為frn, 繪圖資料為Z, 更新圖片plot的Z軸數值, 時間間隔為interal, 單位為ms
ani = animation.FuncAnimation(fig, update, frn, interval=1000/fps)

#plt.show()   # 顯示圖片

ani.save('3D.gif', writer='imagemagick', fps=fps)   # 儲存圖片