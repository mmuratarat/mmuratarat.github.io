---
layout: post
title: "Command Line Arguments"
author: "MMA"
comments: true
---

Command line arguments are flags given to a program/script at runtime. They contain additional information for our program so that it can execute. Not all programs have command line arguments as not all programs need them. Command line arguments allows us to give our program different input on the fly without changing the code. You can draw the analogy that a command line argument is similar to a function parameter. If you know how functions are declared and called in various programming languages, then you’ll immediately feel comfortable when you discover how to use command line arguments.

We must specify shorthand and longhand versions ( -i  and --input ) where either flag could be used in the command line. This is a required argument as is noted by required=True . The help  string will give additional information in the terminal.

`vars` turns the parsed command line arguments into a Python dictionary where the key to the dictionary is the name of the command line argument and the value is value of the dictionary supplied for the command line argument.  Use `print` to see the dictionary.

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