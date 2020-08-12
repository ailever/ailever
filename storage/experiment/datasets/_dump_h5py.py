import h5py

# group save
root = h5py.File('./dataset.hdf5', 'w')
group1 = root.create_group("group1")
group2 = root.create_group("group2")
group3 = root.create_group("group3")

group1_1 = group1.create_group("group1_1")
group1_2 = group1.create_group("group1_2")
group1_3 = group1.create_group("group1_3")

group1_1_1 = group1_1.create_group("group1_1_1")
group1_1_2 = group1_1.create_group("group1_1_2")
group1_1_3 = group1_1.create_group("group1_1_3")


# group load
group1 = root[u'/group1']
group2 = root[u'/group2']
group3 = root[u'/group3']

group1_1 = root[u'/group1/group1_1']
group1_2 = root[u'/group1/group1_2']
group1_3 = root[u'/group1/group1_3']

group1_1_1 = root[u'/group1/group1_1/group1_1_1']
group1_1_2 = root[u'/group1/group1_1/group1_1_2']
group1_1_3 = root[u'/group1/group1_1/group1_1_3']


# dataset save
group1.create_dataset('a', data=list(range(100)))
group1['b'] = list(range(100))
group1['c'] = list(range(200))
group1['d'] = list(range(300))

