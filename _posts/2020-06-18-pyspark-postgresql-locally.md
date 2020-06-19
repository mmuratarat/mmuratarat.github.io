---
layout: post
title: "Using PySpark to connect to PostgreSQL locally"
author: "MMA"
comments: true
---

Apache Spark is a unified open-source analytics engine for large-scale data processing a distributed environment, which supports a wide array of programming languages, such as Java, Python, and R, eventhough it is built on Scala programming language. [PySpark](https://spark.apache.org/docs/latest/api/python/index.html){:target="_blank"} is simply the Python API for Spark that allows you to use an easy programming language, like Python, and leverage the power of Apache Spark.

Spark should be running on a distributed computing system. However, one might not have access to any distributed system all the time. Specially, for learning purpose one might want tor run spark on his/her own computer. This is actually a very easy task to do. 

While working on our local physical machine, we use the built-in standalone cluster scheduler in the local mode (Spark is also deployed through other cluster managers, such as Standalone Cluster Manager, Hadoop YARN, Apache Mesos and Kubernetes). This means that all the Spark processes are run within the same JVM (Java Virtual Machine) effectively. The entire processing is done on a single multithreaded server (technically your local machine is a server). The local mode is very used for prototyping, development, debugging, and testing. However, You can still benefit from parallelisation across all the cores in your server (for example, when you define `.master("local[4]")` in your `SparkSession`, you run it locally with 4 cores), but not across several servers.

For this tutorial, I am using PySpark on Ubuntu Virtual Machine installed on my MacOS and running it on Jupyterlab (in order to use Jupyterlab or Jupyter Notebook, you need `findspark` library. However, you can also use interactive Spark Shell where `SparkSession` has already been created for you). 

You can see my version of Linux below:

```bash
murat@murat-VirtualBox:~$ cat /etc/os-release
NAME="Ubuntu"
VERSION="18.04.4 LTS (Bionic Beaver)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 18.04.4 LTS"
VERSION_ID="18.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=bionic
UBUNTU_CODENAME=bionic
```

Installing Apache Spark on any operating system can be a bit cumbersome but there is a lot of tutorials online. If you follow the steps, you will be just fine.

# How to install PostgreSQL and pgAdmin4 on Linux

PostgreSQL is a free and open source cross-platform Relational Database Management System (RDBMS). It is widely used by developers in the development environment and in production environments as well. Ubuntu’s default repositories contain PostgreSQL packages, so you can install these using the `apt` packaging system. [This tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-18-04){:target="_blank"} explains it so well!

Now, let's deal with pgAdmin4...

pgAdmin4 is free and open-source PostgreSQL graphical user interface for day to day operations with the database. It now comes as native linux package so it is easy to install. You can go to [PostgreSQL site](https://www.postgresql.org/download/linux/ubuntu/){:target="_blank"} and select your Ubuntu version as "Ubuntu Bionic" and run the command below on shell:

```bash
sudo apt-get update && sudo apt-get install pgadmin4
```

in order to install pgAdmin4 on your system. Now we both have locally installed PostgreSQL database and its GUI pgAdmin4. Go ahead and locate pgAdmin4 application and launch it. You will see a new tab open on your browser and Voila! It is working! Let's create some tables!

# Creating a Server in the pgAdmin Dashboard

Before creating tables, first, we need to create a new server. From the pgAdmin dashboard, locate the Browser menu on the left-hand side of the window. Right-click on Servers to open a context menu, hover your mouse over Create, and click Server….

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_pgadmin4.png?raw=true)

This will cause a window to pop up in your browser in which you’ll enter info about your server, role, and database.

In the General tab, enter the name for this server. This can be anything you’d like, but you may find it helpful to make it something descriptive. In our example, the server is named `murat_server`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/2_pgadmin4.png?raw=true)

Next, click on the Connection tab. In the Host name/address field, enter `localhost`. The Port should be set to `5432` by default, which will work for this setup, as that’s the default port used by PostgreSQL.

In the Maintenance database field, enter the name of the database you’d like to connect to. Note that this database must already be created on your server. Then, enter the PostgreSQL username and password you configured while installing pgAdmin4 (or you can create another user if you want). The empty fields in the other tabs are optional, and it’s only necessary that you fill them in if you have a specific setup in mind in which they’re required. Click the Save button, and the database will appear under the Servers in the Browser menu.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3_pgadmin4.png?raw=true)

You’ve successfully connected pgAdmin4 to your PostgreSQL database. You can do just about anything from the pgAdmin dashboard that you would from the PostgreSQL prompt. 

Similarly, we will create a new Database named `database_example`:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/4_pgadmin4.png?raw=true)

# Creating a Table in the pgAdmin

Now, let's create two toy tables, `Employee` and `Department`. 

From the pgAdmin dashboard, locate the Browser menu on the left-hand side of the window. Click on the plus sign (`+`) next to Servers (1) to expand the tree menu within it. Next, click the plus sign to the left of the server you added in the previous step (`murat_server` in our example), then expand Databases, the name of the database you added (`database_example`, in our example), and then Schemas (1). Right-click the Tables list item, then hover your cursor over Query Tool. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/5_pgadmin4.png?raw=true)

This will open a Query Editor, where you can write your own queries to create some tables. Copy and paste the query below and run it:

```sql
DROP TABLE IF EXISTS Employee CASCADE;
DROP TABLE IF EXISTS Department;

Create table If Not Exists Employee (Id int, Name varchar(255), Salary int, DepartmentId int);
Create table If Not Exists Department (Id int, Name varchar(255));

insert into Employee (Id, Name, Salary, DepartmentId) values ('1', 'Joe', '70000', '1');
insert into Employee (Id, Name, Salary, DepartmentId) values ('2', 'Jim', '90000', '1');
insert into Employee (Id, Name, Salary, DepartmentId) values ('3', 'Henry', '80000', '2');
insert into Employee (Id, Name, Salary, DepartmentId) values ('4', 'Sam', '60000', '2');
insert into Employee (Id, Name, Salary, DepartmentId) values ('5', 'Max', '90000', '1');

insert into Department (Id, Name) values ('1', 'IT');
insert into Department (Id, Name) values ('2', 'Sales');
```

Now, tables are created in our database, `database_example` in our server, `murat_server`. Let's connect a locally deployed PostgreSQL database to Spark on JupyterLab. Therefore, we need to download proper PostgreSQL JDBC Driver jar from [https://jdbc.postgresql.org/download.html](this link) depending on Java version installed in your system! Note that you need to enter the correct path to this jar file!

If you want to read in one single table, you can do it multiple ways:

```python
from pyspark.sql import SparkSession

# the Spark session should be instantiated as follows
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.jars", "postgresql-42.2.14.jar") \
    .getOrCreate()
    
 # generally we also put `.master()` argument to define the cluster manager
 # it sets the Spark master URL to connect to, such as “local” to run locally, “local[4]” to run locally with 4 cores, or
 # “spark://master:7077” to run on a Spark standalone cluster.
 # http://spark.apache.org/docs/latest/submitting-applications.html#master-urls
    
# Note: JDBC loading and saving can be achieved via either the load/save or jdbc methods

jdbcDF = spark.read.format("jdbc"). \
options(
         url='jdbc:postgresql://localhost:5432/database_example', # jdbc:postgresql://<host>:<port>/<database>
         dbtable='Employee',
         user='postgres',
         password='1234',
         driver='org.postgresql.Driver').\
load()
# will return DataFrame

jdbcDF2 = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/database_example")\
    .option("dbtable", "Employee") \
    .option("user", "postgres") \
    .option("password", "1234") \
    .option("driver", "org.postgresql.Driver") \
    .load()
# will return DataFrame

jdbcDF3 = spark.read \
    .jdbc("jdbc:postgresql://localhost:5432/database_example", "public.bonus",
          properties={"user": "postgres", "password": "1234", "driver": 'org.postgresql.Driver'})
# will return DataFrame

jdbcDF.printSchema()
# root
#  |-- id: integer (nullable = true)
#  |-- name: string (nullable = true)
#  |-- salary: integer (nullable = true)
#  |-- departmentid: integer (nullable = true)

jdbcDF2.printSchema()
# root
#  |-- id: integer (nullable = true)
#  |-- name: string (nullable = true)
#  |-- salary: integer (nullable = true)
#  |-- departmentid: integer (nullable = true)

jdbcDF3.printSchema()
# root
#  |-- id: integer (nullable = true)
#  |-- name: string (nullable = true)
#  |-- salary: integer (nullable = true)
#  |-- departmentid: integer (nullable = true)

jdbcDF.select('*').collect()
# [Row(id=1, name='Joe', salary=70000, departmentid=1),
#  Row(id=2, name='Jim', salary=90000, departmentid=1),
#  Row(id=3, name='Henry', salary=80000, departmentid=2),
#  Row(id=4, name='Sam', salary=60000, departmentid=2),
#  Row(id=5, name='Max', salary=90000, departmentid=1)]

jdbcDF2.select('*').collect()
# [Row(id=1, name='Joe', salary=70000, departmentid=1),
#  Row(id=2, name='Jim', salary=90000, departmentid=1),
#  Row(id=3, name='Henry', salary=80000, departmentid=2),
#  Row(id=4, name='Sam', salary=60000, departmentid=2),
#  Row(id=5, name='Max', salary=90000, departmentid=1)]

jdbcDF3.select('*').collect()
# [Row(id=1, name='Joe', salary=70000, departmentid=1),
#  Row(id=2, name='Jim', salary=90000, departmentid=1),
#  Row(id=3, name='Henry', salary=80000, departmentid=2),
#  Row(id=4, name='Sam', salary=60000, departmentid=2),
#  Row(id=5, name='Max', salary=90000, departmentid=1)]
```

# How to use a subquery in JDBC data source?

You can also use Spark to process some data from a JDBC source. But to begin with, instead of reading original tables from JDBC, you can run some queries on the JDBC side to filter columns and join tables, and load the query result as a table in Spark SQL.

```python
df = spark.read.jdbc(url = "jdbc:postgresql://localhost:5432/database_example", 
                     table = "(SELECT Employee.id, Employee.name AS employee_name, Employee.salary, Employee.departmentid, Department.name AS department_name \
                     FROM Employee INNER JOIN Department ON Employee.DepartmentId = Department.ID) AS my_table",
                     properties={"user": "postgres", "password": "1234", "driver": 'org.postgresql.Driver'}).createTempView('tbl')

# without createTempView('tbl'), this command will return a DataFrame
```

You can now use SQL commands to fetch any result you want:

```python
spark.sql('select * from tbl').show() #or use .collect() to get Rows

# +---+-------------+------+------------+---------------+
# | id|employee_name|salary|departmentid|department_name|
# +---+-------------+------+------------+---------------+
# |  1|          Joe| 70000|           1|             IT|
# |  2|          Jim| 90000|           1|             IT|
# |  3|        Henry| 80000|           2|          Sales|
# |  4|          Sam| 60000|           2|          Sales|
# |  5|          Max| 90000|           1|             IT|
# +---+-------------+------+------------+---------------+
```

# How to fetch multiple tables using spark sql

In order to fetch multiple tables from PostgreSQL into Spark environment, you need somehow to acquire the list of the tables you have in postgreSQL, which is similarly to what's given below:

```python
#list of the tables in the server
table_names = spark.read.format('jdbc'). \
     options(
         url='jdbc:postgresql://localhost:5432/', # database url (local, remote)
         dbtable='information_schema.tables',
         user='postgres',
         password='1234',
         driver='org.postgresql.Driver'). \
     load().\
filter("table_schema = 'public'").select("table_name")
#DataFrame[table_name: string]

# table_names_list.collect()
# [Row(table_name='employee'), Row(table_name='department')]

table_names_list = [row.table_name for row in table_names.collect()]
print(table_names_list)
# ['employee', 'department']
```

Then, assuming you can create a list of tablenames in python, i.e., `table_names_list`, you can simply call all the tables to Spark to do some manipulations:

```python
url = "jdbc:mysql://localhost:3306/dbname"
reader = sqlContext.read.format("jdbc").option("url",url).option("user","root").option("password", "root")
for tablename in table_names_list:
    reader.option("dbtable",tablename).load().createTempView(tablename)
```

Just be careful though... This will create a temporary view with the same `tablename`. If you want separate names for each tables, you can probably change the initial `table_names_list` with a list of tuple `(tablename_in_sql, tablename_in_spark)`.
