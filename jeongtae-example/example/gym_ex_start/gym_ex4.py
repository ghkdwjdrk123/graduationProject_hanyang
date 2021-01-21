from gym import spaces
space = spaces.Discrete(8)
x = space.sample()
assert space.contains(x)
assert space.n == 8
