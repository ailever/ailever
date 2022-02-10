
## Tabular Analysis
### Regression
`NN Module`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics
```
`Subclassing API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from sklearn.model_selection import train_test_split

class Model(models.Model):
    def __init__(self, name=None):
        super(Model, self).__init__(name)
        self.layer1 = layers.Dense(10, activation='relu')
        self.layer2 = layers.Dense(10, activation='relu')
        self.layer3 = layers.Dense(2, activation='linear')

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# [Dataset]
X = tf.random.normal(shape=(1000, 10))
y = tf.random.normal(shape=(1000, 2))
train_X, test_X, train_y, test_y = train_test_split(X.numpy(), y.numpy(), test_size=0.3, shuffle=False)
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).repeat(10).batch(10)
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(1)

# [Modeling Instances]
model = Model()
criterion = losses.MeanSquaredError()
optimizer = optimizers.Adam(0.1)
train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

# [Train Step]
@tf.function
def train_step(features, targets):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        cost = criterion(targets, predictions)

    gradients = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(cost)
    
# [Test Step]
@tf.function
def test_step(features, targets):
    predictions = model(features, training=False)
    cost = criterion(targets, predictions)
    test_loss(cost)
    
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()

    for features, targets in train_dataset:
        train_step(features, targets)

    for features, targets in test_dataset:
        test_step(features, targets)

    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
    )

model.summary()
```
`Sequential API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics
```
`Functional API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics
```



### Classification
#### Binary Classification
`NN Module`
```python
```
`Subclassing API`
```python
```
`Sequential API`
```python
```
`Functional API`
```python
```

#### Multi-Class Classification
`NN Module`
```python
```
`Subclassing API`
```python
```
`Sequential API`
```python
```
`Functional API`
```python
```

#### Multi-Label Classification
`NN Module`
```python
```
`Subclassing API`
```python
```
`Sequential API`
```python
```
`Functional API`
```python
```


---

## Computer Vision
`NN Module`
```python
```
`Subclassing API`
```python
```
`Sequential API`
```python
```
`Functional API`
```python
```

---

## Natural Language Processing
`NN Module`
```python
```
`Subclassing API`
```python
```
`Sequential API`
```python
```
`Functional API`
```python
```

---

## Time Series Analysis
`NN Module`
```python
```
`Subclassing API`
```python
```
`Sequential API`
```python
```
`Functional API`
```python
```

<br><br><br>

