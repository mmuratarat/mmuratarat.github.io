---
layout: post
title:  "How to create virtual environments for different applications?"
author: "MMA"
comments: true
---

Just imagine that you have an application which is fully developed and you do not want to make any changes to the libraries it is using but at the same time you start developing another application which requires the updated versions of those libraries. What will you do ? It is where virtual environments come into play...

A Virtual Environment, put simply, is an isolated isolated working copy of Python which allows you to work on a specific project without worry of affecting other projects. It enables multiple side-by-side installations of Python, one for each project. It doesn’t actually install separate copies of Python, but it does provide a clever way to keep different project environments isolated because it isolates environmental variables and packages. It is the cure to error messages that complain about the wrong python package version.

Here are some popular libraries/tools for you to create virtual environment in Python: `virtualenv`, `virtualenvwrapper`, `pvenv` and `venv`. Here, we are going to focus on `virtualenv` package.

It is highly likely that that `virtualenv` is already installed on your system. However, check if you have it or which version you are using:

```shell
which virtualenv
```
or 

```shell
virtualenv --version
```

If not, you need to install the package `virtualenv` package:

```shell
(base) Arat-MacBook-Pro:~ mustafamuratarat$ pip3 install virtualenv
Collecting virtualenv
  Downloading virtualenv-20.0.20-py2.py3-none-any.whl (4.7 MB)
     |████████████████████████████████| 4.7 MB 262 kB/s
Collecting distlib<1,>=0.3.0
  Downloading distlib-0.3.0.zip (571 kB)
     |████████████████████████████████| 571 kB 363 kB/s
Collecting appdirs<2,>=1.4.3
  Downloading appdirs-1.4.3-py2.py3-none-any.whl (12 kB)
Requirement already satisfied: six<2,>=1.9.0 in /Users/mustafamuratarat/anaconda3/lib/python3.7/site-packages (from virtualenv) (1.12.0)
Requirement already satisfied: importlib-metadata<2,>=0.12; python_version < "3.8" in /Users/mustafamuratarat/anaconda3/lib/python3.7/site-packages (from virtualenv) (1.6.0)
Requirement already satisfied: filelock<4,>=3.0.0 in /Users/mustafamuratarat/anaconda3/lib/python3.7/site-packages (from virtualenv) (3.0.12)
Requirement already satisfied: zipp>=0.5 in /Users/mustafamuratarat/anaconda3/lib/python3.7/site-packages (from importlib-metadata<2,>=0.12; python_version < "3.8"->virtualenv) (0.5.1)
Building wheels for collected packages: distlib
  Building wheel for distlib (setup.py) ... done
  Created wheel for distlib: filename=distlib-0.3.0-py3-none-any.whl size=340428 sha256=fc418a7494feb2b03cf1be1b75fc5ba1edca92d03922c1ef275af55d309e0506
  Stored in directory: /Users/mustafamuratarat/Library/Caches/pip/wheels/a2/19/da/a15d4e2bedf3062c739b190d5cb5b7b2ecfbccb6b0d93c861b
Successfully built distlib
Installing collected packages: distlib, appdirs, virtualenv
Successfully installed appdirs-1.4.3 distlib-0.3.0 virtualenv-20.0.20
```

Once that is done, go to the folder where your application will reside in and do what's given below, which will call the program `virtualenv` we installed, and it is going to create a folder called `venv` insider our current folder and in `venv` it is going put a fresh Python 3.7.3 installation (you can of course change the Python version you want to install). 

```shell
(base) Arat-MacBook-Pro:REST APIs with Flask and Python mustafamuratarat$ virtualenv venv --python=python3.7.3
```

The command above creates a `venv/` directory in your project where all dependencies are installed. You need to activate your isolated environment first though (in every terminal instance where you are working on your project):

```shell
(base) Arat-MacBook-Pro:REST APIs with Flask and Python mustafamuratarat$ source venv/bin/activate
(venv) (base) Arat-MacBook-Pro:REST APIs with Flask and Python mustafamuratarat$
```

This will fire up a shell (as you see the shell is now on `venv` not `base`) for your environment.

Packages installed here will not affect the global Python installation. `Virtualenv` does not create every file needed to get a whole new python environment It uses links to global environment files instead in order to save disk space end speed up your `virtualenv`. Therefore, there must already have an active python environment installed on your
system.

To install packages for your isolated environment, you can create a text file which include all the packages/tools needed with their versions and do:

```shell
pip3 install -r requirements.txt 
```

In order to deactivate the current environment you can type:

```shell
deactivate
```

You can verify quickly you are in the environment by running `which python3` or `which pip3` which will return the path of the python executable in the environment if all went well:

```shell
(venv) (base) Arat-MacBook-Pro:REST APIs with Flask and Python mustafamuratarat$ which pip3
/Users/mustafamuratarat/Desktop/REST APIs with Flask and Python/venv/bin/pip3
(venv) (base) Arat-MacBook-Pro:REST APIs with Flask and Python mustafamuratarat$ which python3
/Users/mustafamuratarat/Desktop/REST APIs with Flask and Python/venv/bin/python3
```

To remove an environment, make sure you have deactivated it, then cd into the environment directory and type

```shell
sudo rm -rf <my_env_name>
```
