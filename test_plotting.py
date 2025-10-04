"""
测试损失绘图功能的示例脚本
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def test_plotting_functionality():
    """测试matplotlib是否正常工作"""
    
    # 创建测试目录
    save_dir = "./test_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成测试数据
    epochs = range(100)
    loss1 = [10 * np.exp(-i/30) + np.random.normal(0, 0.1) for i in epochs]
    loss2 = [5 * np.exp(-i/25) + np.random.normal(0, 0.05) for i in epochs]
    loss3 = [2 * np.exp(-i/20) + np.random.normal(0, 0.02) for i in epochs]
    
    # 测试基本绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss1, label='Loss 1', marker='o', markersize=2)
    plt.plot(epochs, loss2, label='Loss 2', marker='s', markersize=2)
    plt.plot(epochs, loss3, label='Loss 3', marker='^', markersize=2)
    
    plt.title('Test Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, f'test_plot_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"测试绘图保存到: {save_dir}/test_plot_{timestamp}.png")
    
    # 测试子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(epochs, loss1)
    ax1.set_title('Loss 1')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, loss2)
    ax2.set_title('Loss 2') 
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, loss3)
    ax3.set_title('Loss 3')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(epochs, loss1, label='Loss 1')
    ax4.plot(epochs, loss2, label='Loss 2') 
    ax4.plot(epochs, loss3, label='Loss 3')
    ax4.set_title('All Losses')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'test_subplots_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"测试子图保存到: {save_dir}/test_subplots_{timestamp}.png")
    print("绘图功能测试完成!")

if __name__ == "__main__":
    test_plotting_functionality()