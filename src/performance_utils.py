import numpy as np
import matplotlib.pyplot as plt

def report_computational_cost(times):
    times = np.array(times)
    avg = np.mean(times)
    fps = 1000 / avg
    print("\n📊 Classification Timing Stats:")
    print(f"Frames Processed: {len(times)}")
    print(f"Average Inference Time: {avg:.2f} ms")
    print(f"Min Time: {np.min(times):.2f} ms")
    print(f"Max Time: {np.max(times):.2f} ms")
    print(f"Standard Deviation: {np.std(times):.2f} ms")
    print(f"Estimated FPS: {fps:.1f}")
    return avg

def plot_distribution(times, avg):
    plt.figure(figsize=(10, 4))
    plt.hist(times, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(avg, color="red", linestyle="--", label=f"Avg: {avg:.2f} ms")
    plt.title("Inference Time Distribution per Frame")
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Number of Frames")
    plt.legend()
    plt.tight_layout()
    plt.show()
