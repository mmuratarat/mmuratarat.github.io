---
layout: post
title:  "Bash Shell Tutorial"
author: "MMA"
comments: true
---

# What is a shell?

Traditionally, when you log into a Unix system, the system would start one program for you. That program is a shell, i.e., a program designed to start other programs. It's a command line shell: you start another program by typing its name. 

Simply put, the shell is a a command language interpreter that takes commands in English from the keyboard and gives them to the operating system to perform. Shell is not part of system kernel, but uses the system kernel to execute programs, create files etc. It provides an interface to an Operating System. The shell is only one layer above the OS,

In the old days, it was the only user interface available on a Unix-like system such as Linux. Nowadays, most users prefer the graphical user interface (GUI) offered by operating systems such as Windows, Linux and macOS. Most current Unix-based systems offer both a command line interface (CLI) such as the shell and a graphical user interface.

Two well-known shells are Windows shell and Bash for Linux and macOS.

Shells may be used interactively or non-interactively. As the term implies, interactive means that the commands are run with user-interaction from keyboard. e.g. the shell can prompt the user to enter input from the keyboard. When executing non-interactively, shells execute commands read from a file.

# What is a bash?

Bash is a type of interpreter that processes shell commands. A shell interpreter takes commands in plain text format and calls Operating System services to do something.

It stands for **Bourne Again Shell**, an enhanced version of the original Unix shell program, `sh`, written by Steve Bourne and it is the default shell on many Linux distributions today. Besides `bash`, there are other shell programs that can be installed in a Linux system. These include: Korn shell (`ksh`), enhanced C shell (`tcsh`), friendly interactive shell (`fish`) and `zsh`. Note that each shell does the same job, but each understand a different command syntax and provides different built-in functions.

`bash` and `sh` are two different shells. Basically `bash` is `sh`, with more features and better syntax. Most commands work the same, but they are different.

Bash is one of the popular *command-line* shells, programs whose main job is to start other programs (in addition to some auxiliary functions). The *command-line* part means you control it by typing commands one line at a time. Properly speaking, a graphical user interface (GUI) you use to start programs by double-clicking on icons is also a shell, but in practice by "shell" people mostly mean command-line ones.

# What's a "Terminal?"

Terminal is a program that provides a graphical interface between the shell and the user. It receives from the shell e.g. the characters "command not found" and figures out how to display them to you - with what font, where on the screen, in what colour, whether there should be a scrollbar. When you press some keys, it figures out whether to send them on to the shell as characters (e.g. `ls -l`), or to interpret them on its own (e.g. `⌘C`).

When you open the Terminal app, it automatically opens a shell to connect you to. In its settings, you could choose a different shell from Bash. 

# What is a command?

A "command" is a line that tells the terminal to perform an action. For example:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ date
Mon Mar 30 08:20:02 EDT 2020
```

Another example: `echo` perform an action to print hellp on screen

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo "hello"
hello
```

Another example: `say` will read whatever string you write on terminal.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ say "hello there"
```

There are three rules of commands:
1. A command can act alone. For example, `date` can act alone, you do not need to write anything else.
2. A command can act on something. For example, `echo` command can act on a string. It does not act alone.
3. A command can have **Options** which let the command perform its action in a different way. For example, using command `date` giving time in UTC time zone:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ date -u
Mon Mar 30 12:25:10 UTC 2020
```

#### so, how do you type commands?

syntaxt of commands is given in the following:

```
Command (-option) (something)
````

For example,

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo "Hello"
Hello
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo -n "Hello"
Hello(base) Arat-MacBook-Pro:~ mustafamuratarat$ 
```

if you add `-n` option, it will not print in a new line.

A couple of other examples:
* `killall`: For example, turn on Mac browser Safari and on terminal, do `killall Safari`, it will close the Safari application.
* `cal` can act alone or on something.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cal 01 2020
    January 2020      
Su Mo Tu We Th Fr Sa  
          1  2  3  4  
 5  6  7  8  9 10 11  
12 13 14 15 16 17 18  
19 20 21 22 23 24 25  
26 27 28 29 30 31     
```

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cal
     March 2020       
Su Mo Tu We Th Fr Sa  
 1  2  3  4  5  6  7  
 8  9 10 11 12 13 14  
15 16 17 18 19 20 21  
22 23 24 25 26 27 28  
29 30 31             
```

will give directly current date and month which is 30th of March.

# Some common commands

`whoami` will print your username on your computer:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ whoami
mustafamuratarat
```

`pwd` will print the present working directory:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ pwd
/Users/mustafamuratarat
```

`ls` will print what is insider the current folder, standing for "list":

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ ls
\some folders here
```

`ls -a` will return all the hidden files. Hidden files usually start with a dot (.).

`ls -l` will print details of the visible files in the current folder.

Let's inspect a file after printing using this command

```
drwxr-xr-x@  4 mustafamuratarat  staff    128 Mar 26 11:45 Zoom
```

* Every file or folder has a name (`Zoom`)
* It will also print when the file last edited or created (`Mar 26 11:45`)
* Every file or folder has a size (`128`)
* Every file or directory belongs to some user (owner) (`mustafamuratarat`)
* Every file or directory is assigned to some group which contains one or more users (`staff`)
* It will also print number of links this file has (`4`)
* Files can be Readable (r), Writable (w) or Executable (Accessable) (x) (`drwxr-xr-x@`).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/viewing_permissions.png?raw=true)

Let's look at the first character. The first character identifies the file type: `d` means directory which means a folder. `-` means that this line is a regular file. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/file_types.png?raw=true)

The rest of this line is divided by into three groups. They describe the permissions on the file. The first three `rwx`, the second three `r-x` and the last three `r-x`. The first three characters refer to the owner of the file. They are telling us the owner (`mustafamuratarat`) can read, write and execute the file. The second three characters refer to the group which this folder is assigned to. They are telling us that this folder is the group stuff can read and execute. They cannot write because there is no `w`. The third group of three characters refer to everyone else (or world). They can read and execute but cannot write. 

Sometimes there are the three dashes (---) meaning that they have no permissions at all regarding the file. There’s
a dash where the r normally appears, so they cannot even read it. The dashes afterward tell you they cannot write to the file or execute it. If they try to do anything with the file, they will get a "permission denied" error.

Let's keep on...

`clear` will clear the screen of the terminal. 

You can also see what's inside of a folder without going to that folder by just assigning the path. My home directory has `Desktop` folder. Without changing the directory, I can see what's insider the desktop folder using the command `ls Desktop/`.

I can also find some information about a file using `file` command:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ file X_test.txt
X_test.txt: ASCII text, with very long lines

(base) Arat-MacBook-Pro:~ mustafamuratarat$ file projection_from_2d_into_line.png
projection_from_2d_into_line.png: PNG image data, 1404 x 692, 8-bit/color RGBA, non-interlaced
```

We can change the current directory using the command `cd`.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ pwd
/Users/mustafamuratarat
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cd Desktop/
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ 
```

In order to move out of the current directory, use `cd ..`. We will go back out of present directory one-step.

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ pwd
/Users/mustafamuratarat/Desktop
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ cd ..
(base) Arat-MacBook-Pro:~ mustafamuratarat$ 
```

However, whichever folder you are in, you can just type `cd` and it will take you to the home directory.

Let's say I am in the folder called `spark_book` which is insider `spark` folder which is in `Desktop` folder.

```console
(base) Arat-MacBook-Pro:spark_book mustafamuratarat$ pwd
/Users/mustafamuratarat/Desktop/spark/spark_book
(base) Arat-MacBook-Pro:spark_book mustafamuratarat$ cd 
(base) Arat-MacBook-Pro:~ mustafamuratarat$ pwd
/Users/mustafamuratarat
```

`cd` will directly take me back to home directory which is `/Users/mustafamuratarat`.

Again, we are in the folder `Desktop/spark/spark_book`. If we want to go back two-steps back which is `Desktop` folder but not to home directory:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cd Desktop/spark/spark_book
(base) Arat-MacBook-Pro:spark_book mustafamuratarat$ cd ../..
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ 
```

`open` will open whichever file you call. 

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ open test.txt
```

will open the pdf file `test.txtf`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_command.png?raw=true)

`touch` command will create a file. You will provide the extension.

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ touch Untitled_test.xls
```

will create an Excel file named `Untitled_test`.


`mkdir` will create a new folder, stands for "make directory".

First let's create a new folder named `newfolder` and inside this folder, let's create some files and folders.

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ mkdir newfolder
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ cd newfolder/
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ pwd
/Users/mustafamuratarat/Desktop/newfolder
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch 1.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch 2.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch 3.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch a.txt b.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mkdir A-folder B-folder C-folder
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/2_command.png?raw=true)


What if you want to create a new directory and, at the same time, create a new directory to contain it? 

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ mkdir newfolder2/newfolder
mkdir: newfolder2: No such file or directory
```

Only `mkdir` will not work because `newfolder2` does not even exist.  Simply use the `-p` command option. The following command will create a new folder called `newfolder2` and, at the same time, create a directory within `newfolder2` called `newfolder`:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ mkdir -p newfolder2/newfolder
```

`mv` will move a file (or a folder) or quickly rename it.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv a.txt aaa.txt
```

will rename the file `a.txt` to `aaa.txt`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3_command.png?raw=true)

Similarly,

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv A-folder A-renamed-folder
```

will rename the folder (directory) `A-folder` as `A-renamed-folder`.

Additionally, the command below

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv a.txt A-folder
```

will move the file `aaa.txt` from `newfolder` to `A-folder`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/4_command.png?raw=true)

`cp` will make copy of the existing file.

We are in folder `newfolder`. Let's add some text in `1.txt` using `nano`.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ nano 1.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat 1.txt
write something
```

Let's make a copy of `1.txt` to another file `1a.txt`. 

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cp 1.txt 1a.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat 1a.txt
write something
```

This will result in two identical files: one called `1.txt`and one called `1a.txt`.

(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cp a.html A-folder/

will copy the file `a.html` into the folder `A-folder`.

`rm` command will remove a file. 

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ rm 2.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ rm a.html
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ rm A-folder/a.html
```

The first two commands will remove the files `2.txt` and `a.html` from folder `newfolder` and the last command will remove `a.html` insider the folder `A-folder`.

In some instances,you’ll be asked to confirm the deletion after you issue the command. If you want to delete a file without being asked to confirm it, type the following: `rm –f myfile`.

The `-f` stands for force (that is, force the deletion).

`*` practically means zero or more characters.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ file 1.txt 2.txt
1.txt: ASCII text
2.txt: cannot open `2.txt' (No such file or directory)
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ file 1.txt 3.txt
1.txt: ASCII text
3.txt: empty
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ file *.txt
1.txt:  ASCII text
1a.txt: ASCII text
3.txt:  empty
b.txt:  empty
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ file 1*
1.txt:  ASCII text
1a.txt: ASCII text
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv *.txt A-folder/
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls A-folder/
1.txt	1a.txt	3.txt	aaa.txt	b.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv A-folder/*.txt .
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls A-folder/
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cp b* B-folder/
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls B-folder/
b.html	b.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ rm b*
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls
1.txt		3.txt		B-folder	aaa.txt
1a.txt		A-folder	C-folder
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ rm B-folder/*
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls B-folder/
```

`rm -rf` will remove a directory. 

So, you can type `rm –f *` to delete all files within a directory, or type `rm –f myfile*` to delete all files that start with the word `myfile`. But be careful with the `rm` command. Keep in mind that you cannot salvage files easily if you accidentally delete them!

One important command-line option is `-r`. This stands for *recursive* and tells BASH that you want to compy a directory and its contents (as well as any directories within this directory). Only a handful of BASH commands default to recursive copying. `rm` command needs `-R` option if there are some files inside the folder to be removed. Similar case is valid for `cp`. However, `mv` command does not need such an option. 

For example, `ls -R .` will recursively list all the files inside the working directory and also, files of other folders inside the current directory.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -R .
1.txt		3.txt		B-folder	aaa.txt
1a.txt		A-folder	C-folder

./A-folder:

./B-folder:

./C-folder:
```

`A-folder`, `B-folder` and `C-folder` is empty.

**_WORKING WITH FILES WITH SPACES IN THEM:_**

If, at the command prompt, you try to copy, move or otherwise manipulate files that have spaces in their names, you’ll run into problems. For example, suppose you want to move the file `picture from
germany.jpg` to the directory mydirectory. In theory the following command should do the trick:

```console
mv picture from germany.jpg mydirectory/
```
 
But you will get the following error:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv picture from germany.jpg B-folder
mv: rename picture to B-folder/picture: No such file or directory
mv: rename from to B-folder/from: No such file or directory
mv: rename germany.jpg to B-folder/germany.jpg: No such file or directory
```

There are two solutions. The easiest is to enclose the filename in quotation marks ("), so the previous command would read as follows:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv "picture from germany.jpg" B-folder
```

The other solution is to precede each space with a backslash. This tells BASH you’re including a literal character in the filename. In other words, you’re telling BASH not to interpret the space in the way it normally does, which is as a separator between filenames or commands. Here’s how the command looks if you use backslashes:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv picture\ from\ germany.jpg B-folder
```

# stdin, stdout, and stderr

* **Standard input** - this is the file descriptor that your process reads to get information from you.

* **Standard output** - your process writes normal information to this file descriptor.

* **Standard error** - your process writes errors or log messages to this file descriptor.

These values below are always used for `stdin`, `stdout`, and `stderr`:

* 0: stdin
* 1: stdout
* 2: stderr

# redirect

`echo` will print whatever string it is given. You can redirect this string into a new file.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo "adjnnjkjkfd"
adjnnjkjkfd
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo "hello there" > newtext.txt
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cat newtext.txt
hello there
```

But if you want to add something else in the same file that already exists, 

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo "something else" > newtext.txt
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cat newtext.txt
something else
```

Using `>` command will erase everything what was inside. What if instead we want to append some line into already existing file? In this case we use `>>`.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo "I want to add this line" >> newtext.txt
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cat newtext.txt
something else
I want to add this line
```

You can use this command with every other commands.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ ls -l > listing_home.txt
```

This command redirects the stdout of `ls -l` to a file. If the file already existed, it is overwritten.

NOTE: `>` command is the same as `1>` which means that "redirect all stdout to a file". This `1` is just the file descriptor for `stdout`. The syntax for redirecting is `[FILE_DESCRIPTOR]>`, leaving the file descriptor out is just a shortcut to `1>`.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ ls -l 1> listing_home.txt
```

because `1` stands for `stdout`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/7_command.png?raw=true)


So, to redirect `stderr`, it should be just a matter of adding the right file descriptor in place, which is `2>`. Additionally, when you use `2>&1` you are basically saying "Redirect the `stderr` to the same place we are redirecting the `stdout`". For example:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch foo.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ open foo.txt
```

Here, let's add some text inside `foo.txt`.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat foo.txt
foo
bar
baz
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat foo.txt > output.txt 2>&1
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat output.txt 
foo
bar
baz
```

Note that we don’t see any output in the screen after running `cat foo.txt > output.txt 2>&1` because we use `2>&1`. However, let's do the same for a file `nop.txt` that does not exist in the directory: 

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat nop.txt > output.txt 2>&1
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat output.txt
cat: nop.txt: No such file or directory
```

`output.txt` contains the error messages because we we are redirecting the standard error to `output.txt` too.

Note that `>file 2>&1` is semantically equivalent to `&>` and `>&`.

`>>` is the same as `>` but if the file `listing_home.txt` already existed, `stdout` will be appended to the end of the file instead of overwriting it. 

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ ls -l . >> listing_home.txt
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cat listing_home.txt
total 128496
-rw-r--r--@   1 mustafamuratarat  staff   1238452 May 31  2019 1.png
-rw-r--r--    1 mustafamuratarat  staff    174672 Oct 28 14:20 1_.ipynb
-rw-r--r--@   1 mustafamuratarat  staff    112567 Nov 20 10:44 1_tkx0_wwQ2JT7pSlTeg4yzg.png
-rw-r--r--@   1 mustafamuratarat  staff     13845 May 31  2019 2-229.png
...
```

Let's say we have two text files and we can concatenate them.

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ cat test.txt
Hello!

This is an example file!

Let's open it!

(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ cat test2.txt
This is a new file! 

(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ cat test.txt test2.txt
Hello!

This is an example file!

Let's open it!

This is a new file! 
```

We can redirect this output to a new txt file.

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ cat test.txt test2.txt > new3.txt
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/5_command.png?raw=true)

When you run a sequence of commands in the interactive shell, like

```console
echo xxx; cat file; ls; echo yyy
```

then everything is executed consecutively and the output is send to the terminal.


Let's say `newfile.txt` is a text file in our directory and has `Hello!` string in it. The command below

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ (echo "Some text to prepend"; cat newfile.txt) > file.txt
```

will output 

```console
Some text to prepend
Hello!
```

and this output will be written into `file.txt`.

The direction is not always from left to right. It can also be from right to left!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat < file.txt 
Some text to prepend
Hello!
```

`<` takes the standard input from the file on the right and inputs it into the program on the left.

# pipe

A pipe is a form of redirection (transfer of standard output to some other destination) that is used in Linux and other Unix-like operating systems to send the output of one command/program/process to another command/program/process for further processing. The Unix/Linux systems allow `stdout` of a command to be connected to `stdin` of another command. You can make it do so by using the pipe character `|`. Its syntax is:

```console
command_1 | command_2 | command_3 | .... | command_N 
```

**LESS command**:
`less` is a command line utility that displays the contents of a file or a command output, one page at a time. It is similar to `more`, but has more advanced features and allows you to navigate both forward and backward through the file.

When starting less doesn’t read the entire file which results in much faster load times compared to text editors like `vim` or `nano`. The less command is mostly used for opening large files.

The general syntax for the `less` program is as follows:

```console
less [OPTIONS] filename
```

You can also redirect the output from a command to `less` using a pipe. 

```console
(base) Arat-MacBook-Pro:Desktop mustafamuratarat$ ls -la | less
```

will redirect the contents of the directory `Desktop` to command `less`.

# find

`find` command is a command to look for some files you are interested in.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ find Desktop/newfolder -name aaa.txt
Desktop/newfolder/A-folder/aaa.txt
```

Let's say the folder `Desktop/newfolder/` has some files in it: `aaa.txt` and `aaa_image.png`:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ touch Desktop/newfolder/A-folder/aaa_image.png
(base) Arat-MacBook-Pro:~ mustafamuratarat$ find Desktop/ -name aaa*
Desktop//newfolder/A-folder/aaa_image.png
Desktop//newfolder/A-folder/aaa.txt
```

will find files whose name starts with `aaa`.

Additionally,

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ find Desktop/newfolder -type d
Desktop/newfolder
Desktop/newfolder/B-folder
Desktop/newfolder/C-folder
Desktop/newfolder/A-folder
```

will find all the folders in a folder.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ find Desktop/newfolder -type f
Desktop/newfolder/1a.txt
Desktop/newfolder/.DS_Store
Desktop/newfolder/B-folder/picture from germany.jpg
Desktop/newfolder/3.txt
Desktop/newfolder/1.txt
Desktop/newfolder/A-folder/aaa_image.png
Desktop/newfolder/A-folder/aaa.txt
```

will find all the files in a folder.

# grep

The `grep` filter searches a file for a particular pattern of characters, and displays all lines that contain that pattern.
Syntax:

```console
grep [options] pattern [files]
```

```console
Options Description
-c : This prints only a count of the lines that match a pattern
-h : Display the matched lines, but do not display the filenames.
-i : Ignores, case for matching
-l : Displays list of a filenames only.
-n : Display the matched lines and their line numbers.
-v : This prints out all the lines that do not matches the pattern
-e exp : Specifies expression with this option. Can use multiple times.
-f file : Takes patterns from file, one per line.
-E : Treats pattern as an extended regular expression (ERE)
-w : Match whole word
-o : Print only the matched parts of a matching line,
 with each such part on a separate output line.
```

Let's create a new file.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ cd Desktop/newfolder
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch grep_example.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ nano grep_example.txt 
```
 
Let's add something inside of this text file.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ cat grep_example.txt 
This is a new file.
Something
----

son

A test file.

3.txt
9.txt
```
 
Let's search for `so` inside this file.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ grep so grep_example.txt 
son
```
 
It will pick up the line `son` because grep is case-sensitive. The `-i` option enables to search for a string case insensitively in the give file.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ grep -i so grep_example.txt 
Something
son
```
 
It will also pick up every pattern wherever they are

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ grep s grep_example.txt 
This is a new file.
son
A test file.
```

You can pipe and use the `grep` command.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls
1.txt			3.txt			B-folder		grep_example.txt
1a.txt			A-folder		C-folder
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls | grep t
1.txt
1a.txt
3.txt
grep_example.txt
```

will grep the character `t` in the contents of the directory

You can also print out all the lines that do not matches the pattern using the option `-v`.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls
1.txt			3.txt			B-folder		grep_example.txt
1a.txt			A-folder		C-folder
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls | grep -v t
A-folder
B-folder
C-folder
```

# sudo command

The `sudo` command allows you to run programs with the security privileges of another user (by default, as the superuser). It prompts you for your personal password and confirms your request to execute a command by checking a file, called `sudoers`, which the system administrator configures. Using the `sudoers` file, system administrators can give certain users or groups access to some or all commands without those users having to know the root password. It also logs all commands and arguments so there is a record of who used it for what, and when.

To use the `sudo` command, at the command prompt, enter:

```console
sudo command
```

Replace command with the command for which you want to use`sudo`.

Let's create a new file:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ sudo touch newfile.txt
Password:

(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -la
total 56
drwxr-xr-x  12 mustafamuratarat  staff   384 Mar 30 13:18 .
drwx------@ 59 mustafamuratarat  staff  1888 Mar 30 13:20 ..
-rw-r--r--@  1 mustafamuratarat  staff  8196 Mar 30 10:41 .DS_Store
-rw-r--r--   1 mustafamuratarat  staff    16 Mar 30 09:31 1.txt
-rw-r--r--   1 mustafamuratarat  staff    16 Mar 30 09:32 1a.txt
-rw-r--r--   1 mustafamuratarat  staff     0 Mar 30 09:17 3.txt
drwxr-xr-x   4 mustafamuratarat  staff   128 Mar 30 13:18 A-folder
drwxr-xr-x   3 mustafamuratarat  staff    96 Mar 30 11:16 B-folder
drwxr-xr-x   2 mustafamuratarat  staff    64 Mar 30 09:18 C-folder
-rw-r--r--   1 mustafamuratarat  staff   246 Mar 30 13:00 awk_command_example.txt
-rw-r--r--   1 mustafamuratarat  staff    67 Mar 30 12:51 grep_example.txt
-rw-r--r--   1 root              staff     0 Mar 30 16:07 newfile.txt
```

As you can see, `newfile.txt` will be owned by `root`. Everthing done by `sudo` command will be owned by `root`, superuser.

If we try to edit this file without using `sudo` command, we will get permission error!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/6_command.png?raw=true)

We can also open whole new bash as root!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ sudo bash

The default interactive shell is now zsh.
To update your account to use zsh, please run `chsh -s /bin/zsh`.
For more details, please visit https://support.apple.com/kb/HT208050.
bash-3.2# whoami
root
bash-3.2# rm newfile.txt
bash-3.2# exit
exit
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ whoami
mustafamuratarat
```

# Changing the ownership

Let's say we have a file `dish.txt`, whose ownership belongs to some other user or to root.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ sudo touch dish.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -l
total 32
-rw-r--r--  1 mustafamuratarat  staff   16 Mar 30 09:31 1.txt
-rw-r--r--  1 mustafamuratarat  staff   16 Mar 30 09:32 1a.txt
-rw-r--r--  1 mustafamuratarat  staff    0 Mar 30 09:17 3.txt
drwxr-xr-x  4 mustafamuratarat  staff  128 Mar 30 13:18 A-folder
drwxr-xr-x  3 mustafamuratarat  staff   96 Mar 30 11:16 B-folder
drwxr-xr-x  2 mustafamuratarat  staff   64 Mar 30 09:18 C-folder
-rw-r--r--  1 mustafamuratarat  staff  246 Mar 30 13:00 awk_command_example.txt
-rw-r--r--  1 root              staff    0 Mar 30 16:13 dish.txt
-rw-r--r--  1 mustafamuratarat  staff   67 Mar 30 12:51 grep_example.txt
```

We can change its ownership.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ sudo chown mustafamuratarat dish.txt 
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -l
total 32
-rw-r--r--  1 mustafamuratarat  staff   16 Mar 30 09:31 1.txt
-rw-r--r--  1 mustafamuratarat  staff   16 Mar 30 09:32 1a.txt
-rw-r--r--  1 mustafamuratarat  staff    0 Mar 30 09:17 3.txt
drwxr-xr-x  4 mustafamuratarat  staff  128 Mar 30 13:18 A-folder
drwxr-xr-x  3 mustafamuratarat  staff   96 Mar 30 11:16 B-folder
drwxr-xr-x  2 mustafamuratarat  staff   64 Mar 30 09:18 C-folder
-rw-r--r--  1 mustafamuratarat  staff  246 Mar 30 13:00 awk_command_example.txt
-rw-r--r--  1 mustafamuratarat  staff    0 Mar 30 16:13 dish.txt
-rw-r--r--  1 mustafamuratarat  staff   67 Mar 30 12:51 grep_example.txt
```

As you can see now it belongs to the user `mustafamuratarat`.

We can also change the ownership to a different group.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ sudo chgrp _guest dish.txt 
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -l
total 40
-rw-r--r--  1 mustafamuratarat  staff    16 Mar 30 09:31 1.txt
-rw-r--r--  1 mustafamuratarat  staff    16 Mar 30 09:32 1a.txt
-rw-r--r--  1 mustafamuratarat  staff     0 Mar 30 09:17 3.txt
drwxr-xr-x  4 mustafamuratarat  staff   128 Mar 30 13:18 A-folder
drwxr-xr-x  3 mustafamuratarat  staff    96 Mar 30 11:16 B-folder
drwxr-xr-x  2 mustafamuratarat  staff    64 Mar 30 09:18 C-folder
-rw-r--r--  1 mustafamuratarat  staff   246 Mar 30 13:00 awk_command_example.txt
-rw-r--r--  1 mustafamuratarat  _guest    0 Mar 30 16:13 dish.txt
-rw-r--r--  1 mustafamuratarat  staff    67 Mar 30 12:51 grep_example.txt
```

# Altering permissions of files

You can easily change permissions of files and directories by using the `chmod` command.

If you specify `u`, you can change permissions just for the owner (`u` is for "user"", which is the same as "owner""). You can substitute a `g` to change group (guests) permissions. If you use `a`, it means all users including the owner, the group, and everybody else. Using an `o`, which is for "others"", will configure the file permissions for those who aren't the owner of the file or who are not in the group that owns the file—the last three digits of the permission list.

```console
chmod a+rw myfile
```

In other words, you’re adding read and write (rw) permissions for all users (a), including the owner, the group, and everybody else. Here’s another example:

```console
chmod a-w myfile
```

This tells Linux that you want to take away (-) the ability of all users (a) to write (w) to the file. However, you want to leave the other permissions as they are. 

If you leave out the `a`, `chmod` assumes you mean "all". In other words, commands like `chmod a+r myfile` and `chmod +r` myfile do the same thing. 

```console
chmod u+rw
```

This will add (+) read/write (rw) permissions for the owner.

```console
chmod g-rw
```

This will configure the file so that members of the group that owns the file can’t read or write to it.

The access permission can be specified in the following format. The three parts of the format are given with no spaces between them.

General syntax is given by

```console
[who] [operator] [permission]
```

* Who can be any combination of:
  * u :user (the file's owner).
  * g :group (the file's group members).
  * o :others (everyone else).
  
* Operator can be one of:
  * + :add the permissions to the file's existing set.
  * - :remove the given permissions from the file's set.
  * = :set the permissions to exactly this.
  
* Permission can be any combination of:
  * r :read permission.
  * w :write permission.
  * x :execute permission for programs.

# Variables

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo "this is a line"
this is a line
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ myvar=553
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo "$myvar"
553
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo "The value of my var is $myvar"
The value of my var is 553
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ c=cat
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo "I have $myvar $c"
I have 553 cat
```

Be careful with whitespaces. For a variable assignment, a contiguous string that contains `=` is important. The following will fail:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ myvar = 553
-bash: myvar: command not found
```

In this case, bash splits the input `myvar = 553` into three "words" (`myvar`, `=` and `553`) and then attempts to execute the first word as a command. This clearly is not what was intended here.

We can unset the variable using command `unset` command:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ unset myvar
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo $myvar

(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ 
```

You can also use curly brackets to access to a variable:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ var=hello
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo ${var}
hello
```

This can also be used for commands. Let's say you invent a new command:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mycommand=ls
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo $mycommand
ls
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ $mycommand
1.txt			3.txt			B-folder		awk_command_example.txt	grep_example.txt	newfile.txt.save
1a.txt			A-folder		C-folder		dish.txt		newfile.txt
```

Variable `mycommand` will act like command `ls`.

There are already some variables saved in the system. These are called environment variables.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo $USER
mustafamuratarat
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo $HOME
/Users/mustafamuratarat
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ECHO $PATH
/Users/mustafamuratarat/anaconda3/bin:/Users/mustafamuratarat/anaconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.6/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```

If you run the `printenv` or `env` command without any arguments it will show a list of all environment variables:

You can also use `read` command to read a variable.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ read myvariable
Hellohello there32
(base) Arat-MacBook-Pro:~ mustafamuratarat$ echo $myvariable
Hellohello there32
```

# alias

```console
alias alias_name="command_to_run"
```

The `alias` command allows you to create keyboard shortcuts, or aliases, for commonly used commands. They are typically placed in the `~/.bash_profile` or `~/.bashrc` files.


Open the `~/.bash_profile` in your text editor:


```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ nano ~/.bash_profile
```

and add your aliases:

```console
# Aliases
# alias alias_name="command_to_run"
```

The aliases should be named in a way that is easy to remember. It is also recommended to add a comment for future reference.

Once done, save and close the file. Make the aliases available in your current session by typing:

```console
source ~/.bash_profile
```

`source` activates the changes in `~/.bash_profile` for the current session. Instead of closing the terminal and needing to start a new session, `source` makes the changes available right away in the session we are in.

# Shell script

The terminal usually allows just one command at the time. Shell scripts allow you to combine and run multiple commands together. A shell scripting is writing a program for the shell to execute and a shell script is a file or program that shell will execute. Shell scripts also allow you to use if-else statements and loops. 

Let's create a new file called `shortscript.txt`.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch shortscript.txt
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ nano shortscript.txt 
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -l shortscript.txt 
-rw-r--r--  1 mustafamuratarat  staff  0 Apr  1 06:54 shortscript.txt
```

Let's make it executable for all the users (we leave out the `a`, `chmod` assumes you mean "all")

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ chmod +x shortscript.txt 
```

and add

```console
#!/bin/bash

echo Hello
echo bye
```

in this file.

In order to run this executable file, we can just do,

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ file shortscript.txt 
shortscript.txt: Bourne-Again shell script text executable, ASCII text
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$  ./shortscript.txt 
Hello
bye
```

# Difference Between #!/bin/sh and #!/bin/bash

As stated previously, `bash` and `sh` are two different shells. Basically `bash` is `sh`, with more features and better syntax. Most commands work the same, but they are different.

`bash` binary is stored in the `/bin/bash` path in general. `/bin/sh` provides the `sh` shell which is cirppeled down version of the `bash`.

We can use `which` command to see where our `bash` lives in:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ which bash
/bin/bash
```

# which

`which` command in Linux is a command which is used to locate the executable file associated with the given command by searching it in the path environment variable.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ which ls
/bin/ls
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ which pwd
/bin/pwd
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ which python
/Users/mustafamuratarat/anaconda3/bin/python
```

`info which`: It displays help information.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/8_command.png?raw=true)

You can see this `bin` folder and locate all the built-in executable files:

```console
(base) Arat-MacBook-Pro:/ mustafamuratarat$ ls
Applications	Preboot 1	Volumes		dev		opt		tmp
Library		System		bin		etc		private		usr
Preboot		Users		cores		home		sbin		var
(base) Arat-MacBook-Pro:/ mustafamuratarat$ open bin
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/9_command.png?raw=true)

Even by clicking on these files, you can run those executables. So, when you run `ls` command on your command window, you are actually running `/bin/ls`.

```console
(base) Arat-MacBook-Pro:/ mustafamuratarat$ /bin/ls
Applications	Preboot 1	Volumes		dev		opt		tmp
Library		System		bin		etc		private		usr
Preboot		Users		cores		home		sbin		var
(base) Arat-MacBook-Pro:/ mustafamuratarat$ /bin/pwd
/
```

# Creating the first script

If you constantly run the same set of commands at the command line, why not automate that?

Let's create a file first!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch myscript.txt
```

and then we add on the first line of the program,

```console
#!
```

which is known as "Shebang".

Shebang contains path (of executable), from which you are telling Interpreter that execute following code from given executable path (followed by `#!`).

Since we are going to use Shell language, `bash`, the path is 

```console
#/bin/bash
```

Let's add some commands in this file!

```console
#! /bin/bash

echo hello this is my first script

#this is a comment
#it does not affect the program
```

When we try to run it, we will get permission denied error.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./myscript.txt
-bash: ./myscript.txt: Permission denied
```

Because it is not executable! When we give necessary permissions and run it, it will work!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ chmod +x myscript.txt 
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$  ./myscript.txt
hello this is my first script
```

However, `.txt` is not a proper extension for an executable file! For shell file, an appropriate extension is `.sh`. Let's change it and run!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ mv myscript.txt myscript.sh
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./myscript.sh 
hello this is my first script
```

# Changing the path

The information about where your programs are stored, and therefore where Ubuntu should look for commands you type in, as well as any programs you might want to run, is stored in the `PATH` variable. You can take a look at what’s currently stored there by typing the following:

```console
echo $PATH
```

The `echo` command merely tells the shell to print something on screen. In this case, you are telling it to "echo"" the PATH variable onto your screen. 

Several directories are in this list, each separated by a colon. 

The important thing to know is that whenever you type a program name, the shell looks in each of the listed directories in
sequence. In other words, when you type `ls`, the shell will look in each of the directories stored in the `PATH` variable, starting with the first in the list, to see if the `ls` program can be found. The first instance it finds is the one it will run.

Let's say you have a executable file `myscript.sh` inside a folder `/Users/mustafamuratarat/Desktop/newfolder`. 

```console
#! /bin/bash

echo "What is your name?"
read yourname

echo "hello $yourname ! nice to meet you!"


#this is a comment
#it does not affect the program
```

Everytime you run it, it will ask you to enter your name!

If you are already in the directory where the program in question is located, you can type the following:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ myscript.sh
```

But what if you want to run a program that is not contained in a directory listed in your `PATH`? In this case, you must tell the shell exactly where the program is. 


You have to enter the full path of the script with whichever directory you are in, i.e., `/Users/mustafamuratarat/Desktop/newfolder/myscript.sh`. If you do not do this, you will get "command not found" error!

Let's say you are on home directory and the script that you want to run is in another folder:

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ myscript.sh
-bash: myscript.sh: command not found

(base) Arat-MacBook-Pro:~ mustafamuratarat$ /Users/mustafamuratarat/Desktop/newfolder/myscript.sh
What is your name?
MMA
hello MMA ! nice to meet you!
```

In order to prevent it, you can add the PATH to `.bash_profile`.

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ nano .bash_profile
```

Add the line below

```console
PATH="/Users/mustafamuratarat/Desktop/newfolder:${PATH}"
export PATH
```

to `.bash_profile` and source it. 

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ source .bash_profile
```

Now, you can run the script everywhere and even use `tab` to complete its name!

```console
(base) Arat-MacBook-Pro:~ mustafamuratarat$ myscript.sh 
```

or 

```console
(base) Arat-MacBook-Pro:MLE_BOOK mustafamuratarat$ myscript.sh 
```

or 

```console
(base) Arat-MacBook-Pro:paper1 mustafamuratarat$ myscript.sh 
```

et cetera... Pay attention that I can access the script from different directories!

### A small example: create a script from another script

Let's first create a new `.sh` file and open it

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch create_script.sh
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ open create_script.sh 
```

and add the lines below into this `create_script.sh` file:

```console
#! /bin/bash

read -p "name of the script to create: " name_s

touch ${name_s}

echo "#! /bin/bash " >> ${name_s}
echo "######## AUTOMATICALLY CREATED ######## " >> ${name_s}

chmod +x ${name_s}

echo "DONE!"
```

Let's give execution permission to this scrip

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ chmod +x create_script.sh
```

and then let's run it!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./create_script.sh
name of the script to create: my_try.sh
DONE!
```

It will ask to provide a name to this new script and will automatically give permissions will add two lines of codes.

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -l
total 80
-rwxr-xr-x@ 1 mustafamuratarat  staff   210 Apr  1 07:59 create_script.sh
-rwxr-xr-x  1 mustafamuratarat  staff    57 Apr  1 08:00 my_try.sh
```

As you can see, `my_try.sh` is already executable!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/10_command.png?raw=true)

# How to to echo a blank line in a shell script?

All of these commands can be used to echo a blank line:

`echo`, `echo ''`, `echo ""`

We cant use `echo "\n"` or `echo '\n'` as they will give output as `\n` in both cases.


# Arithmetic Expressions

(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo "$((2+3))"
5

Let's create a script which uses some mathematical operations:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ touch arithmetic_expressions.sh
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ open arithmetic_expressions.sh 
```

and write the script given below to this shell script:

```console
#! /bin/bash

number1=10
number2=20
echo
echo "number1 is $number1 and number2 is $number2"
echo
echo "SUM $((number1+number2))"
echo "PRODUCT $((number1*number2))"
echo "DIVISION $((number2/number1))"
echo "REMAINDER $((number1%number2))"

echo '----------------------------------'

echo "3^2 is"
echo "POWER $((3**2))"

echo '----------------------------------'

echo "variable is $number1"
echo "$((number1++))" #it will print the number first and then increase it!
echo "variable now is $number1" #so it will print 11 here

echo '----------------------------------'

echo "value=$number1"
echo "Let's add 3 to number1"
echo "ADD: $((number1+=3))"
```

Don't you forget to give execution permission `chmod +x arithmetic_expressions.sh`.

When you run this script, the output will be:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./arithmetic_expressions.sh 
 
number1 is 10 and number2 is 20

SUM 30
PRODUCT 200
DIVISION 2
REMAINDER 10
----------------------------------
3^2 is
POWER 9
----------------------------------
variable is 10
10
variable now is 11
----------------------------------
value=11
Let's add 3 to number1
ADD: 14
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ 
```

# if-else statement

```shell
if [ expression1 ]
then
   statement1
   statement2
   .
   .
elif [ expression2 ]
then
   statement3
   statement4
   .
   .
else
   statement5
fi
```

# Numeric comparison operators

some operations are given below:

```console
3 -eq 3  	3 = 3
3 -ne 4		3 is not 4
3 -gt 1		3 > 1
3 -lt 7		3 < 7

3 -ge 3 		3 >= 3
3 -le 3 		3 <= 3
```

some examples are given below:

```console
#! /bin/bash

echo "hello"

read -p "how old are you?    " age

if [ $age -gt 100 ]; then
    echo "you are not very young"
else
    echo "you are still very young"
fi

echo "bye"


####################################

read -p "Type a integer number between 1 and 4: " num

if [ $num == "1" ]; then
    echo "typed 1"
    elif [ $num == "2"  ]; then
        echo "typed 2"
    elif [ $num == "3"  ]; then
        echo "typed 3"
    elif [ $num == "4"  ]; then
        echo "typed 4"
    else
        echo "none of the above"

fi
```

# exit

You can use the `exit` statement to terminate shell script upon an error. Most of the time it is being used a parameter `n`. `n` is the return of status (also called exit status). If `n` is not provided, then it simply returns the status of last command that is executed.

As a rule, most commands return an exit status of 0 if they were successful, and 1 if they were unsuccessful (it contains minor errors).

`echo $?` returns the status of the last finished command. Status 0 tells you that everything finished ok.

Every command has a exit status in its description page. For example, for `pwd`:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/11_command.png?raw=true)

exit status is particularly important in shell scripting because you can take some actions in case you get some problem with some commands that you are running.

# what does `$` mean in a shell script?

![](https://i.stack.imgur.com/90t48.png)

* The `$` character represents the process ID number, or PID, of the current shell −
  ```console
  (base) Arat-MacBook-Pro:newfolder mustafamuratarat$ echo $$
  6087
  ```
* `$0` returns the filename of the current script. Put it in the script `echo "File Name: $0"`.

# logic conditions

```console
-a AND
-o OR
```

```console
#! /bin/bash

echo "hello"
read -p "How old are you?  " age

#using OR condition
if [ $age -lt 0 -o $age -gt 200 ]; then
    echo "Number Not Acceptable"
    exit
fi

#using AND condition
if [ $age -gt 26 -a $age -lt 64 ]; then
    echo "you are between 26 and 64"
    exit
fi
echo "Ok let's contunue with the script:"
```

# String comparison operators

String comparison operators enable the comparison of alphanumeric strings of characters. 

* `-z string`: True if the length of string is zero.
* `-n string`: True if the length of string is non-zero.
* `string1 == string2`: True if the strings are equal.
* `string1 != string2`:	True if the strings are not equal.
* `string1 < string2`: 	True if string1 sorts before string2 lexicographically (refers to locale-specific sorting sequences for all alphanumeric and special characters).
* `string1 > string2`:	True if string1 sorts after string2 lexicographically.

```console
#! /bin/bash

read -p "Type something:  (Enter to exit)" str

#check if str is empty
if [ -z $str ]; then
 	echo "this is an empty string"
	exit
fi
echo "moving on"
```

# if on files

Let's write a script to see whether a file exists in the directory or not. We use `read` command to enter the name of the file interactively.

```console
#! /bin/bash

# Condition to check if a file EXISTS

read -p "Enter the file name:  " myfile

if [ -e $myfile ]; then
   echo "${myfile} exists!"
fi

# Negate a condition

if [ ! -e $myfile ]; then
   echo "${myfile} does not exist!"
fi


# Condition to check if a file is a DIRECTORY

if [ -d $myfile ]; then
   echo "${myfile} is a directory!"
else
   echo "it is NOT a directory!"
fi

# Condition to check if a file is readable

if [ -r $myfile ]; then
   echo "${myfile} is readable!"
fi


# Condition to check if a file is writable

if [ -w $myfile ]; then
   echo "${myfile} is writable!"
fi


# Condition to check if a file is executable

if [ -x $myfile ]; then
   echo "${myfile} is executable!"
fi

# Condition to check if a file is executable

if [ -s $myfile ]; then
   echo "${myfile} is empty!"
fi
```

Output of this script will be:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
Enter the file name:  file.txt
file.txt exists!
it is NOT a directory!
file.txt is readable!
file.txt is writable!
file.txt is empty!
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
Enter the file name:  file2.txt
file2.txt does not exist!
it is NOT a directory!
```

because `file.txt` exists in `newfolder` folder, it is empty and it is a file NOT directory and it is readable/writable but not executable for the user!

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ls -l file.txt
-rw-r--r--@ 1 mustafamuratarat  staff  29 Mar 31 12:55 file.txt
```

However, `file2.txt` does not exists in the folder and it is not a directory.

# sleep

`sleep` is a command-line utility that allows you to suspends the calling process for a specified time. In other words, the `sleep` command pauses the execution of the next command for a given number of seconds.

The `sleep` command is useful when used within a bash shell script, for example, when retrying a failed operation or inside a loop

The syntax for the `sleep` command is as follows:

```console
sleep NUMBER[SUFFIX]...
```

* The `NUMBER` may be a positive integer or a floating-point number.
* The `SUFFIX` may be one of the following:
  * s - seconds (default)
  * m - minutes
  * h - hours
  * d - days
  * When no suffix is specified, it defaults to seconds.
  
  
When two or more arguments are given, the total amount of time is equivalent to the sum of their values.

Here are a few simple examples demonstrating how to use the `sleep` command:

Sleep for 5 seconds:

```console
sleep 5
```

Sleep for 0.5 seconds:

```console
sleep 0.5
```

Sleep for 2 minute and 30 seconds:

```console
sleep 2m 30s
```

An shell script example:

```console
#! /bin/bash

echo " some lines of code here "

for i in {0..6}
do
    echo "number:    $i "
    sleep 1.5
done

echo " "
echo "bye"
```

will output

```console
 some lines of code here 
number:    0 
number:    1 
number:    2 
number:    3 
number:    4 
number:    5 
number:    6 
 
bye
```

With every iteration, there will be 1.5 seconds pause.

# Loops

```console
#! /bin/bash

echo " some lines of code here"

for i in {1,2,3,4}
do
  echo "Hello, this is number $i"
done

echo "we continue..."
```

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
 some lines of code here
Hello, this is number 1
Hello, this is number 2
Hello, this is number 3
Hello, this is number 4
we continue...
```

here `i` can be anything:

```console
#! /bin/bash

echo " some lines of code here"

for i in {1,"cat", -5, "hello", "something" -98}
do
  echo "Hello, this is number $i"
done

echo "we continue..."
```

will print out

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
 some lines of code here
Hello, this is number {1,cat,
Hello, this is number -5,
Hello, this is number hello,
Hello, this is number something
Hello, this is number -98}
we continue...
```

We can use `break` command to break the loop:

```console
#! /bin/bash

for i in {0,"danger","dog","hello there",9}
do
    echo "this is the value  $i"
    if [ $i == "danger" ]; then
        echo "**** WE have to stop the loop here!!!!!****"
        break
    fi
done
```

will run and break right after `i` equals "danger" and print out:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
this is the value  0
this is the value  danger
**** WE have to stop the loop here!!!!!****
```

We can also go through all the files in a directory using for-loop.

```console
#! /bin/bash

#* stands for 'all'
for i in ./*
do
  echo "name of the file if $i"
done
```

will print 

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
name of the file if ./1.txt
name of the file if ./1a.txt
name of the file if ./3.txt
name of the file if ./A-folder
name of the file if ./B-folder
name of the file if ./C-folder
name of the file if ./arithmetic_expressions.sh
name of the file if ./awk_command_example.txt
name of the file if ./create_script.sh
name of the file if ./dish.txt
name of the file if ./file.txt
name of the file if ./grep_example.txt
name of the file if ./if_else.sh
name of the file if ./my_try.sh
name of the file if ./myscript.sh
name of the file if ./newfile.txt
name of the file if ./shortscript.txt
```
If you want to increment `i` in a loop, you can do it in two different approaches:

```console
#! /bin/bash

#APPROACH 1
for ((i=1;i<=5;i++))
do
  echo "the number is $i"
done

# OR

echo -e "\n or \n"

#APPROACH 2
for i in {0..5}
do
  echo "the number is $i"
done
```

will print out:

```console
(base) Arat-MacBook-Pro:newfolder mustafamuratarat$ ./if_else.sh 
the number is 1
the number is 2
the number is 3
the number is 4
the number is 5

 or 

the number is 0
the number is 1
the number is 2
the number is 3
the number is 4
the number is 5
```

# while loop

The syntax is:

```console
while [ condition ]
do
      command1
      command2
      ..
      ....
      commandN
done
```

An example:

```console
#! /bin/bash

# set n to 1
n=1

# continue until $n equals 5
while [ $n -le 5 ]
do
    echo "Welcome $n times."
    n=$(( n+1 ))     # increments $n
done
```

The script initializes the variable n to 1, and then increments it by one. The while loop prints out the "Welcome $n times" until it equals 5 and exit the loop.

You can use `((expression))` syntax to test arithmetic evaluation (condition) to improve code readability.

```console
#! /bin/bash

# set n to 1
n=1

n=1
while (( $n <= 5 ))
do
    echo "Welcome $n times."
    n=$(( n+1 ))
done
```

# How to Use SCP Command to Securely Transfer Files

SCP (secure copy) is a command-line utility that allows you to securely copy files and directories between two locations.

With `scp`, you can copy a file or directory:

* From your local system to a remote system.
* From a remote system to your local system.
* Between two remote systems from your local system.

When transferring data with `scp`, both the files and password are encrypted, so that anyone snooping on the traffic doesn’t get anything sensitive.

The `scp` command relies on `ssh` for data transfer, so it requires an `ssh` key or password to authenticate on the remote systems. Thus, you need to set it up beforehand.

The colon (`:`) is how `scp` distinguish between local and remote locations.

To be able to copy files you must have at least read permissions on the source file and write permission on the target system.

Be careful when copying files that share the same name and location on both systems, `scp` will overwrite files without warning.

To copy a file from a local to a remote system, run the following command:

```console
scp file.txt remote_username@10.10.0.2:/remote/directory
```

Where, `file.txt` is the name of the file we want to copy, `remote_username` is the user on the remote server, `10.10.0.2` is the server IP address. The `/remote/directory` is the path to the directory you want to copy the file to. If you don’t specify a remote directory, the file will be copied to the remote user home directory.

To copy a directory from a local to remote system, use the `-r` option:

```console
scp -r /local/directory remote_username@10.10.0.2:/remote/directory
```

To copy a file from a remote to a local system, use the remote location as a source and local location as the destination.

For example to copy a file named `file.txt` from a remote server with IP 10.10.0.2 run the following command:

```console
scp remote_username@10.10.0.2:/remote/file.txt /local/directory
```

To copy a directory from remote system to a local directory, use the `-r` option:

```console
scp -r remote_username@10.10.0.2:/remote/directory /local/directory
```

when using scp you don’t have to log in to one of the servers to transfer files from one to another remote machine:

The following command will copy the file `/files/file.txt` from the remote host `host1.com` to the directory `/files` on the remote host `host2.com`.

```
scp user1@host1.com:/files/file.txt user2@host2.com:/files
```

You will be prompted to enter the passwords for both remote accounts. The data will be transfer directly from one remote host to the other.

# What does ampersand (`&`) mean at the end of a shell script line?

When you run your script, you can add `&` at the end of your command line.

This is known as "job control"" under unix. The `&` informs the shell to put the command in the background, so you can continue to use the shell and do not have to wait until the script is finished. If you forget it, you can stop the current running process with `Ctrl-Z` and continue it in the background with `bg` (or in the foreground with `fg`).

You can see the list of jobs presently running with the `jobs` command. 
