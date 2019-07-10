---
layout: post
title: "Command Line Arguments"
author: "MMA"
comments: true
---

[Command line arguments](https://docs.python.org/3/library/argparse.html){:target="_blank"} are flags given to a program/script at runtime. They contain additional information for our program so that it can execute. Not all programs have command line arguments as not all programs need them. Command line arguments allows us to give our program different input on the fly without changing the code. You can draw the analogy that a command line argument is similar to a function parameter. If you know how functions are declared and called in various programming languages, then you’ll immediately feel comfortable when you discover how to use command line arguments.

We must specify shorthand and longhand versions ( -i  and --input ) where either flag could be used in the command line. This is a required argument as is noted by required=True . The `help` string will give additional information in the terminal.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cla_help.png?raw=true)

`vars` turns the parsed command line arguments into a Python dictionary where the key to the dictionary is the name of the command line argument and the value is value of the dictionary supplied for the command line argument.  Use `print` to see the dictionary.

For this simple example, it will be:

{% highlight python %}
print(args)
#{'name': 'Murat'}
{% endhighlight %}

You can then use those arguments in your model!

Let's create a `simple_example.py` file and see how it works!

{% highlight python %}
#import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="name of the user")
args = vars(ap.parse_args())


# display a friendly message to the user
print("Hi there {}, it's nice to meet you!".format(args["name"]))
{% endhighlight %}

Go do it, yourself! You can use either `-n` or `--name` to assign a value to the argument!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cla_simple_example.png?raw=true)


**EDIT**: The `tf.app.flags` module is a functionality provided by Tensorflow to implement command line flags for your Tensorflow program, which Google requires internally for its developers to use. Look [here](https://stackoverflow.com/questions/33932901/whats-the-purpose-of-tf-app-flags-in-tensorflow){:target="_blank"}, [here](https://planspace.org/20170314-command_line_apps_and_tensorflow/){:target="_blank"} and [here](https://abhisheksaurabh1985.github.io/2017-12-30-flags-in-python-tf/){:target="_blank"} for more!
