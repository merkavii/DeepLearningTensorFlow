import numpy as np
import tensorflow as tf


# * Create tensors with tf.constant()
scalar = tf.constant(7)
print('*'* 96)
print(f'Tensor: {scalar} | Number of dimentions: {scalar.ndim} | dtype: {scalar.dtype}')

# * Create a vector
vector = tf.constant([10,10])
print(f'Vector: {vector} | Number of dimentions: {vector.ndim} | dtype: {vector.dtype}')

# * Create a matrix
matrix = tf.constant([[10, 7],
                     [7,10]])
print(f'Matrix: {matrix} | Number of dimentions: {matrix.ndim} | dtype: {matrix.dtype}')
print('*'* 96)

# * Create another matrix
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [6., 9.]], dtype=tf.float16)
print(another_matrix)
print('*****')

tensor = tf.constant([[ [1,2,3],
                        [4,5,6]],
                        [[7,8,9],
                        [10,11,12]],
                        [[13,14,15],
                         [16,17,18]]])
print(tensor)
print(f'Number of dimentions: {tensor.ndim}')
print('*'* 96)

# * Create the same tensor with 
changable_tensor = tf.Variable([10,7])
unchangable_tensor = tf.constant([10,7])
print(changable_tensor)
print(unchangable_tensor)

# ! unchangable_tensor[0].assign(7) -> error

print('*'* 96)
# * Create random tensors
random_1 = tf.random.Generator.from_seed(42) # ? setting seed for reproducibility
random_1 = random_1.normal(shape=(3, 2))

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))

# ? are theu equal?
print(random_1 == random_2)
print('*'* 96)

# * Shuffle a tensor (المنت های داخل تنسور ترتیبشونو عوض میکنیم.جهت بهبود یادگیری مدل)
not_shuffled = tf.constant([[1, 2],
                            [3,4],
                            [5,6]])
shuffled = tf.random.shuffle(not_shuffled) #? فقط در بعد اول شافل شد.یعنی مثلا از 1,2 به 2,1 تبدیل نشد
print(f'{not_shuffled} ------->\n {shuffled}')
print('*'* 96)

# * Create a tensor of all ones and zeros
print(tf.ones([2,3]))
print(tf.zeros(shape=(2,3)))

# * Turning NumPy array into Tensorflow tensors
numpy_A = np.arange(1,25, dtype=np.int32)
A = tf.constant(numpy_A)
B = tf.constant(numpy_A, shape=(4,6)) # ^ 6 * 4 = 24 
print(A)
print(B)

print('*'* 96)
# * Getting information from tensors
# | Shape  Rank  Axis(dimention)  Size

# * Create a rank 4 tensors (4 dimensions)
rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
print(f'Rank 4 tensor: {rank_4_tensor}')
print(
    f'Rank 4 tensor shape: {rank_4_tensor.shape}\n'
    f'Number of dimentions(Rank): {rank_4_tensor.ndim}\n'
    f'Size: {tf.size(rank_4_tensor)}\n'
    f'Datatype: {rank_4_tensor.dtype}\n'
    f'Elements along the 0 axis: {rank_4_tensor.shape[0]}\n'
)
print('*'* 96)

# * Adding extra dimention
rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # ^ or we can use tf.expand_dims(rank_2_tensor, axis=1)
print(f'Shape before expand: {rank_2_tensor.shape}  Shape after expand: {rank_3_tensor.shape}')

print('*'* 96)

# * Manipulating tensors (Tensor operations)
    # @ Basic operations (+ - * /)
tensor = tf.constant([[10, 7],
                    [3, 4]])
print(f'Tensor + 10 = {tensor + 10} or {tf.add(tensor, 10)}')
print(f'Tensor * 10 = {tensor * 10} or {tf.multiply(tensor, 10)}')
print(f'Tensor - 10 = {tensor - 10} or {tf.subtract(tensor, 10)}')
print(f'Tensor / 10 = {tensor / 10} or {tf.divide(tensor, 10)}')

    # @ Matrix multiplciiation (Dot Product)
print(f'Matrix multiplciiation: {tf.matmul(tensor, tensor)}') # ! This is not equal with tensor * tensor
print('*'* 96)

# ? Create a tensor with (3, 2) shape 
X = tf.constant([[1,2],
                 [3,4],
                 [5,6]]) # ^ shape = (3, 2)
# ? Create another tensor with (3, 2) shape 
y = tf.constant([[7,8],
                 [9,10],
                 [11,12]]) # ^ shape = (3, 2)

# ! tf.matmul(X, y) <--- This will be error
# | There are 2 rules in Matrix multiplciiation
    # ^ The inner dimentions must match
    # ^ The resulting matrix has the shape of the outer dimentions

# ? Let's change the shape of y
print(f'Matrix multiplciiation with y reshape: {tf.matmul(X, tf.reshape(y, shape=(2, 3))  )} shape = (3,3)\n------------------')
print(f'Matrix multiplciiation with X reshape: {tf.matmul(tf.reshape(X, shape=(2, 3)), y)} shape = (2,2)')

# ^ We can use tf.transpose but know this that transpose flips the axis but reshape is defferent
print('*'* 96)

print('Normal y:')
print(y)

print('y reshaped to (2, 3):')
print(tf.reshape(y, (2,3)))

print('y transposed:')
print(tf.transpose(y))
print('*'* 96)

# * Changing the datatype of a tensor
# ? Create a new tensor with default datatype (float32,int32)
B = tf.constant([1.7, 7.4])
C = tf.constant([7, 10])
print(f'B dtype: {B.dtype} | C dtype: {C.dtype}')
print(f'New B dtype: {tf.cast(B, dtype=tf.float16).dtype} | New C dtype: {tf.cast(C, dtype=tf.int16).dtype}')

print('*'* 96)
