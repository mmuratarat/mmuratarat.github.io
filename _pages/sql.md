---
layout: post
title: "Frequently Asked Questions (and Answers)"
author: MMA
comments: false
permalink: /faq/
---

#### What is SQL?

SQL stands for Structured Query Language , and it is used to communicate with the Database. This is a standard language used to perform tasks such as retrieval, updation, insertion and deletion of data from a database.

#### What is Database?

Database is nothing but an organized form of data for easy access, storing, retrieval and managing of data. This is also known as structured form of data which can be accessed in many ways.

#### What are the different subsets of SQL?

* DDL (Data Definition Language) – It allows you to perform various operations on the database such as `CREATE`, `ALTER` and `DELETE` objects.
* DML ( Data Manipulation Language) – It allows you to access and manipulate data. It helps you to `INSERT`, `UPDATE`, `DELETE` and retrieve data from the database.
* DCL ( Data Control Language) – It allows you to control access to the database. Example – Grant, Revoke access permissions.

#### What is a query?

A database query is a request for data or information from a database table or combination of tables. A database query can be either a select query or an action query.

#### What is subquery?

A subquery is a query within another query. The outer query is called as main query, and inner query is called subquery, also known as nested query. SubQuery is always executed first and one time, and the result of subquery is passed on to the main query.

#### How to create a new database table?

```sql
CREATE TABLE IF NOT EXISTS mytable (
    column DataType TableConstraint DEFAULT default_value,
    another_column DataType TableConstraint DEFAULT default_value,
    ...
);
```
#### How to insert rows into a table?

```sql
INSERT INTO mytable (column, another_column, ...)
VALUES (value_or_expr, another_value_or_expr, ...),
      (value_or_expr_2, another_value_or_expr_2,  ...),
       ...;
```

#### How to insert rows from one table to another?

The `INSERT INTO SELECT` statement copies rows from one table and inserts it into another table.

```sql
INSERT INTO table2 (column1, column2, column3, ...)
SELECT column1, column2, column3, ...
FROM table1
WHERE condition;
```

In order to use above query, you must have already created `table2`. When `table2` is ready you can insert rows from `table1`. 

If you want to create a new table and insert values into it from another table at the same time, use query below:

```sql
CREATE TABLE table2 AS
SELECT column1, column2, column3, ... FROM table1 WHERE condition;
```

#### How to update rows in a table?

```sql
UPDATE mytable
SET column = value_or_expr, 
    other_column = another_value_or_expr, 
    ...
WHERE condition;
```

#### How to delete rows in a table?

The command below will remove specific records defined by WHERE clause:

```sql
DELETE FROM mytable
WHERE condition;
```

Commit and Rollback can be performed after delete statement. `DELETE` command is a DML (Data Manipulation Language) command.

If you want to delete all the rows:

```sql
DELETE FROM mytable
```

The `TRUNCATE` command also removes all rows of a table. We cannot use a `WHERE` clause in this. This operation cannot be rolled back. This command is a DDL (Data Definition Language) command.

```sql
TRUNCATE TABLE table_name;
```

To remove all data from multiple tables at once, you separate each table by a comma (,) as follows:

```sql
TRUNCATE TABLE table_name1, table_name2, ...
```

#### How to copy a table definition?

You want to create a new table having the same set of columns as an existing table. You do not want to copy the rows, only the column structure of the table:

```sql
CREATE TABLE new_table
AS 
SELECT * FROM existing_table where 1=0
```

The subquery in above query will return no rows.

#### What are the different operators available in SQL?

There are three operators available in SQL, namely:

1. Arithmetic Operators such as `+`, `-`, `*`, `/`, and `%`.
2. Bitwise Operators such as `&`, `|`, and `^`.
3. Comparison Operators such as `=`, `!=`, `>`, `<`, `>=`, `<=` and `<>`.
4. Logical Operators such as `AND`, `NOT`, `OR`, `ANY`, `BETWEEN`, `LIKE`, and `IN`.

#### In which order do SQL queries happen?

Consider the SQL SELECT statement syntax:

{% highlight sql %}
SELECT DISTINCT <TOP_specification> <select_list>
FROM <left_table>
<join_type> JOIN <right_table>
ON <join_condition>
WHERE <where_condition>
GROUP BY <group_by_list>
HAVING <having_condition>
ORDER BY <order_by_list>
{% endhighlight %}

![](https://jvns.ca/images/sql-queries.jpeg)
Source: [https://jvns.ca/blog/2019/10/03/sql-queries-don-t-start-with-select/](https://jvns.ca/blog/2019/10/03/sql-queries-don-t-start-with-select/){:target="_blank"}

the order is:

1. `FROM/JOIN` and all the `ON` conditions
2. `WHERE`
3. `GROUP BY`
4. `HAVING`
5. `SELECT` (including window functions)
6. `DISTINCT`
6. `ORDER BY`
7. `LIMIT`

In practice this order of execution is most likely unchanged from above. With this information, we can fine-tune our queries for speed and performance.

#### How to drop a table?

If a table is dropped, all things associated with the tables are dropped as well. This includes - the relationships defined on the table with other tables, the integrity checks and constraints, access privileges and other grants that the table has. Therefore, this operation cannot be rolled back.

```sql
DROP TABLE IF EXISTS mytable;
```


#### How to add a new column to a table?

```sql
ALTER TABLE mytable
ADD column DataType OptionalTableConstraint DEFAULT default_value;
 ```
 
#### How to remove a column from a table? 

```sql
ALTER TABLE mytable
DROP column_to_be_deleted;
```

#### How to rename a table?

```sql
ALTER TABLE mytable
RENAME TO new_table_name;
```

#### What are the data types in PostgreSQL?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-03%20at%2009.22.09.png?raw=true)

#### What's the difference between VARCHAR and CHAR?

`VARCHAR(x)` is variable-length, which can have up to x characters. `CHAR(x)` is fixed length, which can only have exactly x characters. `CHAR` always uses the same amount of storage space per entry, while `VARCHAR` only uses the amount necessary to store the actual text. If your content is a fixed size, you'll get better performance with `CHAR`.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-12-21%20at%2010.56.33.png?raw=true)

#### What is the default ordering of data using the ORDER BY clause? How could it be changed?

The default sorting order is ascending. It can be changed using the DESC keyword, after the column name in the ORDER BY clause.

#### What are the constraints?

Constraints are additional requirements for acceptable values in addition to those provided by data types. They allow you to define narrower conditions for your data than those found in the general purpose data types.

These are often reflections on the specific characteristics of a field based on additional context provided by your applications. For example, an `age` field might use the `int` data type to store whole numbers. However, certain ranges of acceptable integers do not make sense as valid ages. For instance, negative integers would not be reasonable in this scenario. We can express this logical requirement in PostgreSQL using constraints.

When you create a table, you can create a constraint using the CREATE TABLE command's CONSTRAINT clause. There are two types of constraints: column constraints and table constraints. In other words, postgreSQL allows you to create constraints associated with a specific column or with a table in general.

* Column level constraint is declared at the time of creating a table but table level constraint is created after table is created.
* Composite primary key must be declared at table level.
* All the constraints can be created at table level but for table level NOT NULL is not allowed.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-03%20at%2009.21.57.png?raw=true)

Let's give an example how to define a column constraint and a table constraint. A column constraint is defined by:

```sql
CREATE TABLE person (
    . . .
    age int CHECK (age >= 0),
    . . .
);
```

and table constraint by:

```sql
CREATE TABLE person (
    . . .
    age int,
    . . .
    CHECK (age >= 0)
);
```

Note that multiple column constraints are separated by a space. For instance:

```sql
CREATE TABLE mytable(name CHAR(10) NOT NULL,
        id INTEGER REFERENCES idtable(id),
        age INTEGER CHECK (age > 0));
```

The syntax to add constraints to an existing table column is as follows:

```sql
ALTER TABLE table_name
ADD constaint_type (column_name);
```

To remove (drop) a constraint you need to know its name. If the name is known, it is easy to drop. Else, you need to find out the system-generated name.

```sql
ALTER TABLE table_name DROP CONSTRAINT constraint_name;
```

#### What is CHECK constraint?

You normally use the `CHECK` constraint at the time of creating the table using the `CREATE TABLE` statement. The following statement defines an `employees` table.

```sql
CREATE TABLE employees (
   id serial PRIMARY KEY,
   first_name VARCHAR (50),
   last_name VARCHAR (50),
   birth_date DATE CHECK (birth_date > '1900-01-01'),
   joined_date DATE CHECK (joined_date > birth_date),
   salary numeric CHECK(salary > 0)
);
```

The `employees` table has three `CHECK` constraints:

1. First, the birth date (`birth_date`) of the employee must be greater than "01/01/1900". If you try to insert a birth date before 0"1/01/1900", you will receive an error message.
2. Second, the joined date (`joined_date`) must be greater than the birth date (`birth_date`). This check will prevent from updating invalid dates in terms of their semantic meanings.
3. Third, the salary must be greater than zero, which is obvious.

#### What is a DEFAULT constraint?

A `DEFAULT` constraint is used to include a default value in a column when no value is supplied at the time of inserting a record. For example;

```sql
CREATE TABLE Persons
(
P_Id int NOT NULL,
LastName varchar(255) NOT NULL,
FirstName varchar(255),
Address varchar(255),
City varchar(255) DEFAULT 'Sandnes'
)
```

The SQL code above will create `City` column whose default value is `Sandnes`.

#### What is a UNIQUE constraint?

A `UNIQUE` constraint ensures that all values in a column are different. This provides uniqueness for the column(s) and helps identify each row uniquely. Unlike primary key, there can be multiple unique constraints defined per table.

```sql
CREATE TABLE Students ( 	 /* Create table with a single field as unique */
    ID INT NOT NULL UNIQUE
    Name VARCHAR(255)
);

CREATE TABLE Students ( 	 /* Create table with multiple fields as unique */
    ID INT NOT NULL
    LastName VARCHAR(255)
    FirstName VARCHAR(255) NOT NULL
    CONSTRAINT PK_Student
    UNIQUE (ID, FirstName)
);

ALTER TABLE Students 	 /* Set a column as unique */
ADD UNIQUE (ID);

ALTER TABLE Students 	 /* Set multiple columns as unique */
ADD CONSTRAINT PK_Student 	 /* Naming a unique constraint */
UNIQUE (ID, FirstName);
```

#### What is the difference between primary key and unique constraints?

Primary key cannot have NULL value, the unique constraints can have NULL values. There is only one primary key in a table, but there can be multiple unique constraints. The primary key creates the cluster index automatically but the Unique key does not.

### What is the difference between a relation and a relationship in SQL?

The term relation is sometimes used to refer to a table with columns and rows in a relational database and relationship is association between relations/tables. 

# How to drop all the tables in a PostgreSQL database?

If all of your tables are in a single schema, this approach could work (below code assumes that the name of your schema is `public`):

```
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
```

If you are using PostgreSQL 9.3 or greater, you may also need to restore the default grants.

```
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;
```

Note that this will also delete all functions, views, etc defined in the public schema.

Note that this will not remove the system tables (such as those that begin with `pg_`) as they are in a different schema, `pg_catalog`.


#### What are the different types of relationships in SQL?

* **One-to-One** : This can be defined as the relationship between two tables where each record in one table is associated with the maximum of one record in the other table.
* **One-to-Many** and **Many-to-One** : This is the most commonly used relationship where a record in a table is associated with multiple records in the other table.
* **Many-to-Many** : This is used in cases when multiple instances on both sides are needed for defining a relationship.
* **Self Referencing Relationships** : This is used when a table needs to define a relationship with itself.

#### Which operator is used in query for pattern matching?

LIKE operator is used for pattern matching, and it can be used as -.

* % - Matches zero or more characters.
* \_ (Underscore) – Matching exactly one character.

```sql
Select * from Student where studentname like 'a%'
```

```sql
Select * from Student where studentname like 'ami_'
```

#### What is the difference between BETWEEN and IN operators in SQL?

The BETWEEN operator is used to fetch rows based on a range of values.
For example,

```sql
SELECT * FROM Students WHERE ROLL_NO BETWEEN 20 AND 30;
```
This query will select all those rows from the table Students where the value of the field ROLL_NO lies between 20 and 30.

The IN operator is used to check for values contained in specific sets.
For example,

```sql
SELECT * FROM Students WHERE ROLL_NO IN (20,21,23);
```

This query will select all those rows from the table Students where the value of the field ROLL_NO is either 20 or 21 or 23.

#### What is a primary key and a foreign key?

A primary key is a special database table column or combination of columns (also called Composite PRIMARY KEY) designated to uniquely identify all table records. A primary key's main features are:
1. It must contain a unique value for each row of the data. 
2. It cannot contain null values (it has an implicit NOT NULL constraint).

A table in SQL is strictly restricted to have one and only one primary key, which is comprised of single or multiple fields (columns).

A primary key is either an existing table column or a column that is specifically generated by the database according to a defined sequence.

A foreign key is a column or group of columns in a relational database table that provides a link between data in two tables. It acts as a cross-reference between tables because it references the primary key of another table, thereby, establishing a link between them. A table can have multiple foreign keys.

The table with the foreign key constraint is labelled as the child table, and the table containing the candidate key is labelled as the referenced or parent table.

Depending on what role the foreign key plays in a relation:
1. It can not be NULL if this foreign key is also a key attribute.
2. It can be NULL, if this foreign key is a normal attribute.

It can also contain duplicates. Whether it is unique or not unique relates to whether the table has a one-one or a one-many relationship to the parent table.

An example for relation between primary key and foreign key:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/primary_foregin_keys.png?raw=true)

Note that the orders table contains two keys: one for the order and one for the customer who placed that order. In scenarios when there are multiple keys in a table, the key that refers to the entity being described in that table is called the primary key (PK) and other key is called a foreign key (FK).

In our example, `order_id` is a primary key in the orders table, while `customer_id` is both a primary key in the customers table and a foreign key in the orders table. Primary and foreign keys are essential to describing relations between the tables, and in performing SQL joins.

To add a primary key constraint into an existing table, we use the following syntax:

```sql
ALTER TABLE tablename
ADD PRIMARY KEY (column_name);
```

or

```sql
ALTER TABLE table_name
ADD CONSTRAINT MyPrimaryKey PRIMARY KEY (column1, column2...);
```

The basic syntax of ALTER TABLE to DROP PRIMARY KEY constraint from a table is as follows −

```sql
ALTER TABLE table_name
DROP CONSTRAINT MyPrimaryKey;
```

Foreign keys are added into an existing table using the ALTER TABLE statement. The following syntax is used:

```sql
ALTER TABLE child_table
ADD CONSTRAINT constraint_name FOREIGN KEY (c1) REFERENCES parent_table (p1);
```
In the above syntax, the child_table is the table that will contain the foreign key while the parent table shall have the primary keys. C1 and p1 are the columns from the child_table and the parent_table columns respectively.

#### How to avoid duplicate records in a query?

The `SELECT DISTINCT` query is used to return only unique values. It eliminates all the duplicated values.
