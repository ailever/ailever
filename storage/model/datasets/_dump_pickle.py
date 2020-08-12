import pickle

obj = type('obj', (), {})

OBJ = obj()
setattr(OBJ, 'a', obj())
setattr(OBJ, 'b', obj())
setattr(OBJ, 'c', obj())

pickle.dump(OBJ, open('./dataset.pkl', 'wb'))


