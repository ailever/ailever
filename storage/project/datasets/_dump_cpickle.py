from six.moves import cPickle

obj = type('obj', (), {})

OBJ = obj()
setattr(OBJ, 'a', obj())
setattr(OBJ, 'b', obj())
setattr(OBJ, 'c', obj())

cPickle.dump(OBJ, open('./dataset.pkl', 'wb'))
    
