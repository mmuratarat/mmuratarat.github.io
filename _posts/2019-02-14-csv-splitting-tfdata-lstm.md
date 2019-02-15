---
layout: post
title: "Reading CSV file by using Tensorflow Data API and Splitting Tensor into Training and Test Sets for LSTM"
author: "MMA"
comments: true
---

There might be times when you have your data only in a one huge `CSV` file and you need to feed it into Tensorflow and at the same time, you need to split it into two sets: training and testing. Using [`train_test_split` function of Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} cannot be proper because of using a [`TextLineReader`](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset){:target="_blank"} of [Tensorflow Data API](https://www.tensorflow.org/api_docs/python/tf/data){:target="_blank"} so the data is now a tensor. Scikit-Learn works on Numpy arrays not Tensorflow's tensors.

Here, we will show how you can do it using [Tensorflow Data API](https://www.tensorflow.org/api_docs/python/tf/data){:target="_blank"} efficiently. Firstly, let's create a Pandas's dataframe, consisting of 204 observations, 33 features and one binary labels and save it as a CSV file into our directory.

{% highlight python %}
import numpy as np
import pandas as pd

my_df  = pd.DataFrame(np.random.rand(204,33))
my_df[34] = np.array(np.random.binomial(size=204, n=1, p= 0.7)).T
my_df.to_csv('data.csv', index = False, sep=',', encoding='utf-8', header=False)
print(my_df.shape)
#(204, 34)
{% endhighlight %}

What we are also going to do here is to explain how we can transform this dataset into a structure, where we can use it in a LSTM network. LSTM accepts a structure in the shape of `[batch_size, time_steps, number_of_features]`. Here, batch_size will be kept dynamic. Let's choose `time_steps = 6`. We already know that number of features is 33. Therefore, the rows in CSV file will represent time-steps and every 6 time-steps will denote one person (one customer, one observation, et cetera). Total number of observations is then `204/6 = 34`. 

Let's split the dataset into two and use $80\%$ of it for training and the rest for testing. Therefore, we will have 27 observations (162 rows) for training and 7 observations (42 rows) for testing.

{% highlight python %}
def input_fn(dataset):
    num_steps = 6
    
    # Combine 6 lines into a single observation.   
    dataset = dataset.batch(num_steps)

    def decode_csv(line):
        rDefaults = [[0.]] * 33 + [[0]] 
        parsed_line = tf.decode_csv(line, record_defaults=rDefaults)
        label = parsed_line[-1:] # Last column is the label
        del parsed_line[-1] # Delete last column
        features = parsed_line # Everything (but last column) are the features
        return features, label

    # Parse each observation into a `row_per_ob X 33` matrix of features and a
    # scalar label which will be one-hot encoded.
    dataset = dataset.map(decode_csv)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=lambda *x:(tf.transpose(x[0], [1,0]), tf.cast(tf.squeeze(tf.one_hot(tf.transpose(x[1], [1,0]), num_classes)), tf.int32)), batch_size=batch_size, num_parallel_calls=num_parallel_calls, drop_remainder=False)).prefetch(prefetch_batch_buffer)
    return dataset
{% endhighlight %}

`input_fn()` function will decode the CSV. First, it combines 6 lines into a single observation, then reads 33 float columns for features and one integer column for labels. It also one-hot-encodes the label variable. 

Let's choose `batch_size = 3` which means that every batch will consists of 3 observations (18 rows). Therefore, training set will have 9 total batches without a remainder and total number of batches for testing set will be 3 with the last batch only having one observation (6 rows).

{% highlight python %}
import multiprocessing

batch_size = 3
num_steps = 6
num_classes = 2
num_inputs = 33
prefetch_batch_buffer = 2
number_of_rows = 204
num_parallel_calls = multiprocessing.cpu_count()

file_name = 'data.csv'
dataset = tf.data.TextLineDataset(file_name)

number_of_observations = number_of_rows/num_steps
number_of_rows_training = int(np.floor(number_of_observations*80/100)*num_steps)

train_dataset = input_fn(dataset.take(number_of_rows_training))
test_dataset = input_fn(dataset.skip(number_of_rows_training))

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
X, y = iterator.get_next(name='iterator_op')

training_init_op = iterator.make_initializer(train_dataset, name = 'training_initialization')
testing_init_op = iterator.make_initializer(test_dataset, name = 'testing_initialization')

with tf.Session() as sess:
    print('\nPrinting training batches')
    sess.run(training_init_op)
    total_train_batch = int((number_of_rows_training/(num_steps * batch_size)))
    for _ in range(total_train_batch):
        X_train, y_train = sess.run([X, y])
        print(X_train.shape)
    
    print('\nPrinting test batches')
    sess.run(testing_init_op)
    total_testing_batch = int(((number_of_rows-number_of_rows_training)/(num_steps * batch_size)) + 1)
    for _ in range(total_testing_batch):
        X_test, y_test = sess.run([X, y])
        print(X_test.shape)

# Printing training batches
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)
# (3, 6, 33)

# Printing test batches
# (3, 6, 33)
# (3, 6, 33)
# (1, 6, 33)
{% endhighlight %}

As can be seen easily, here, we are using [`.take()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take){:target="_blank"} and [`.skip()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip) function of Tensorflow data API.

Additionally, we use `Reinitializable Iterator` here so then we switch dynamically between different input data streams. We create an iterator for different datasets. Note that all the datasets must have the same datatype and shape. Do not also forget that iterator has to be initialized before it starts running.
