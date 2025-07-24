import pickle
import jax
import jax.numpy as jnp
import numpy as np

all_x_train, all_y_train, all_x_test, all_y_test = None, None, None, None
with open('./data/cifar100.pkl', 'rb') as f:
    all_x_train, all_y_train, all_x_test, all_y_test = pickle.load(f)

np.random.seed(1993)
class_order = np.random.permutation(100).tolist()
print("Class order:", class_order)

active_classes = class_order[:5]  # first 5 classes for the first task
print("Active classes:", active_classes)
mask = np.isin(all_y_train, active_classes)
x = jnp.transpose(jnp.array(all_x_train[mask]), (0, 2, 3, 1))
y = jnp.array(all_y_train[mask])

x_eval = jnp.transpose(jnp.array(all_x_test[np.isin(all_y_test, active_classes)]), (0, 2, 3, 1))
y_eval = jnp.array(all_y_test[np.isin(all_y_test, active_classes)])

dummy_output = jax.random.normal(jax.random.PRNGKey(0), (x.shape[0], 100))
print("dummy_output shape:", dummy_output.shape)
dummy_output_sliced = dummy_output[:, active_classes]
print("dummy_output_sliced shape:", dummy_output_sliced.shape)

# compute accuracy 
preds = jnp.argmax(dummy_output_sliced, axis=-1)
print("Predictions:", preds)
labels = jnp.array([active_classes.index(int(label)) for label in y])
print("Labels:", labels)
print("Accuracy on the selected classes:", jnp.mean(preds == labels))