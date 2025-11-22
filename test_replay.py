from arlforge.utils.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=5)

buffer.add([1,2,3], 0, 1, [1.1, 2.1, 3.1], False)
buffer.add([4,5,6], 1, 0.8, [4.1, 5.1, 6.1], False)

print("Buffer size:", len(buffer))
print("Sample:", buffer.sample(1))
