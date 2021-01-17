---
layout: post
title:  "Simple Running Average using Window Functions of PostgreSQL"
author: "MMA"
comments: true
---

Recently, a friend of mine discovered a weird but interesting property of `ORDER BY` while using a window function intriqued me. I did not know that and I wanted to share it.

First, let's create two tables named `products` and `product_groups` for the demonstration:

```sql
CREATE TABLE product_groups (
   group_id serial PRIMARY KEY,
   group_name VARCHAR (255) NOT NULL
);
 
CREATE TABLE products (
   product_id serial PRIMARY KEY,
   product_name VARCHAR (255) NOT NULL,
   price DECIMAL (11, 2),
   group_id INT NOT NULL,
   FOREIGN KEY (group_id) REFERENCES product_groups (group_id)
);
```

Second, let's insert some rows into these tables:

```sql
INSERT INTO product_groups (group_name)
VALUES
   ('Smartphone'),
   ('Laptop'),
   ('Tablet');
 
INSERT INTO products (product_name, group_id,price)
VALUES
   ('Microsoft Lumia', 1, 200),
   ('HTC One', 1, 400),
   ('Nexus', 1, 500),
   ('iPhone', 1, 900),
   ('HP Elite', 2, 1200),
   ('Lenovo Thinkpad', 2, 700),
   ('Sony VAIO', 2, 700),
   ('Dell Vostro', 2, 800),
   ('iPad', 3, 700),
   ('Kindle Fire', 3, 150),
   ('Samsung Galaxy Tab', 3, 200);
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-15%20at%2008.14.45.png?raw=true)

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-15%20at%2008.15.16.png?raw=true)

To apply the aggregate function to subsets of rows, we use the GROUP BY clause. The following example returns the average price for every product group.

```sql
SELECT
   group_name,
   TRUNC(AVG (price),2) AS AVERAGE_PRICE
FROM
   products
INNER JOIN product_groups USING (group_id)
GROUP BY
   group_name;
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-15%20at%2008.17.31.png?raw=true)

As you see clearly from the output, the `AVG()` function reduces the number of rows returned by the queries in both examples. Similar to an aggregate function, a window function operates on a set of rows. However, it does not reduce the number of rows returned by the query. The term window describes the set of rows on which the window function operates. A window function returns values from the rows in a window.

```sql
SELECT
   product_name,
   price,
   group_name,
   TRUNC(AVG (price) OVER (
      PARTITION BY group_name), 2) AS Average_Price
FROM
   products
   INNER JOIN 
      product_groups USING (group_id);
```

which will return:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-15%20at%2008.20.05.png?raw=true)

which is what we expect. The products will be grouped by their name, i.e., Laptop, Smartphone, table and the window function will compute the average for each of these product group and place the same average value for each record.

But sometimes, you might need to compute the running average on the same partition, instead of overall average. In this case, `ORDER BY` clause comes to help. It sorts the rows within the window, and processes them in that order including only the current and previously seen rows, ignoring rows that are after current or not in the window.

```sql
SELECT
   product_name,
   price,
   group_name,
   TRUNC(AVG (price) OVER (
      PARTITION BY group_name ORDER BY price), 2) AS Average_Price
FROM
   products
   INNER JOIN 
      product_groups USING (group_id);
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-02-15%20at%2008.28.09.png?raw=true)

As can be seen easily, after grouping by name, in each group, we sort the records by price. The first value of `average_price` column is 700 because there is no other smaller value than this price. The second one also has its value of `average_price` column as 700 because $\frac{700 + 700}{2} = 700$. For the third record, we will have three observations and we compute the mean of 700, 700 and 800 and thereby, we have 733.3. Similarly, the fourth record has 800 for its `average_price` column because the average of 700, 700, 800 and 1,200 is 850 and so on...
