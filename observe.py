import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
    accident=np.load('accident_2015.npy')
    green_taxi=np.load('green_taxi_traffic_2015.npy')
    yellow_taxi=np.load('yellow_taxi_traffic_2015.npy')
    print(accident.shape)
    sum_over_area=accident.sum(-1).sum(-1)
    sum_over_time=accident.sum(0)
    day=100 # select one day and plot accident for the next 15 days
    plt.figure(figsize=(10, 6))
    for i in range(3):
        for j in range(5):
            idx=i*5+j
            plt.subplot(3,5,idx+1)
            y=sum_over_area[day*24+idx*24:day*24+(idx+1)*24]
            x=np.arange(1, 25)
            plt.bar(x, y)
    plt.savefig('histogram_of_accident.png')
    plt.figure(figsize=(12, 8))
    for i in range(4):
        for j in range(6):
            idx=i*6+j
            plt.subplot(4,6,idx+1)
            plt.imshow(accident[day*24+idx,:,:])
    plt.savefig('heatmap_of_accident.png')
    plt.figure(figsize=(12, 8))
    for i in range(4):
        for j in range(6):
            idx=i*6+j
            plt.subplot(4,6,idx+1)
            plt.imshow(green_taxi[day*24+idx,:,:])
    plt.savefig('heatmap_of_green_taxi.png')
    plt.figure(figsize=(12, 8))
    for i in range(4):
        for j in range(6):
            idx=i*6+j
            plt.subplot(4,6,idx+1)
            plt.imshow(yellow_taxi[day*24+idx,:,:])
    plt.savefig('heatmap_of_yellow_taxi.png')

