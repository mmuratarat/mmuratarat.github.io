---
layout: post
title: "Mac-Os Catalina Update — ZSH instead of Bash ('Command not found' issue for Jupyter)"
author: "MMA"
comments: true
---

After the recent update of MacOS, which is named Catalina, the terminal asks you to switch from bash to zsh by running a command. 

ZSH is great and has many improvements over bash including themes and plugins. However you can have problems with running Jupyter notebook or Jupyter Lab with issue thrown away 'command not found' because they are not supported directly.

There might be multiple solutions for this problem but this worked for me.

The `~/.zshrc` file doesn’t exist by default in MacOS X so you need to create it. The `~/` translates to your user’s home directory and the .`zshrc` is the ZSH configuration file itself.

So just open up a “Terminal” and create that file like this; I am using nano as a text editor but feel free to use whatever text editor you feel comfortable with:

> nano ~/.zshrc

One needs to add an alias of here in case — jupyter. Follow the following steps.

> alias jupyter=’/Library/Frameworks/Python.framework/Versions/3.7/bin/jupyter-notebook ‘

Copy the above line to add an alias to the `~/.zshrc` file.

Now to save the file in `nano` just hit <kbd>ctrl</kbd>+<kbd>X</kbd>. When it prompts:

> Save modified buffer (ANSWERING "No" WILL DESTROY CHANGES) ?

Just type "Y" and then you will get a new prompt which looks something like this; just note the path `/Users/mustafamuratarat/` will match your local user’s path:

> File Name to Write: /Users/mustafamuratarat/.zshrc

Now, just hit <kbd>return</kbd> and the file will be saved and you will now be back to the command line prompt in the "Terminal". If you now exit from the Terminal and then open a new window, the `~/.zshrc` settings should now be loaded in.

And voila! Jupyter Notebook runs perfectly. It can similarly be implemented for other applications.

