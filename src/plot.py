import pickle
import numpy as np 
import matplotlib.pyplot as plt
E = 1000

training_rewards = np.zeros((3, E))
with open("rewards_0_param.pickle", 'rb') as fp:
    seed0 = pickle.load(fp)
training_rewards[0,] = seed0
with open("rewards_1_param.pickle", 'rb') as fp:
    seed1 = pickle.load(fp)
training_rewards[1,] = seed1
with open("rewards_2_param.pickle", 'rb') as fp:
    seed2 = pickle.load(fp)
training_rewards[2,] = seed2


avg_training_rewards = np.mean(training_rewards, axis = 0) # get column averages 
print("now plot")
print("avg training rewards", avg_training_rewards)
# avg_training_rewards /= 3
plt.figure(figsize=(12, 8))
# linestyles = ['--', '-.', ':']
for s in range(3):
    plt.plot([i for i in range(E)], training_rewards[s,], label = "Seed %s" % s)
print("ave line")
plt.plot([i for i in range(E)], avg_training_rewards, label = "Average", alpha = .3, color = "k")
plt.legend(loc = "upper left")
plt.xlabel("Episode Num")
plt.ylabel("Average training reward")
plt.title("MountainCarContinuous-v0, Exploration type: RND")
plt.savefig(("MountainCarContinuous-v0_RND") + ".png", dpi = 300)