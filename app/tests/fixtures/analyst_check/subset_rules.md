# SQL you can use

Use only the SQL forms listed under Works below. Anything else may fail or give a wrong answer.

## Works

- count_star: `SELECT COUNT(*) AS order_count FROM orders`
- sum_expr: `SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem WHERE l_shipyear = 1995`
- avg: `SELECT AVG(o_totalprice) AS avg_price FROM orders WHERE o_orderyear = 1996`
- min_max: `SELECT MIN(o_orderdate) AS first_order, MAX(o_orderdate) AS last_order FROM orders`
- group_by: `SELECT o_orderpriority, COUNT(*) AS order_count FROM orders GROUP BY o_orderpriority`
- order_by_column: `SELECT n_name, n_regionkey FROM nation ORDER BY n_name`
- order_by_alias: `SELECT l_shipmode, SUM(l_quantity) AS total_quantity FROM lineitem GROUP BY l_shipmode ORDER BY total_quantity DESC, l_shipmode`
- limit: `SELECT o_orderkey, o_totalprice FROM orders ORDER BY o_totalprice DESC, o_orderkey LIMIT 5`
- table_alias: `SELECT o.o_orderkey, o.o_orderdate FROM orders o WHERE o.o_orderkey <= 7`
- join_on: `SELECT n_name, COUNT(*) AS customer_count FROM customer JOIN nation ON c_nationkey = n_nationkey GROUP BY n_name`
- implicit_join: `SELECT r_name, COUNT(*) AS nation_count FROM nation, region WHERE n_regionkey = r_regionkey GROUP BY r_name`
- three_way_join: `SELECT r_name, SUM(c_acctbal) AS total_balance FROM customer JOIN nation ON c_nationkey = n_nationkey JOIN region ON n_regionkey = r_regionkey GROUP BY r_name`
- between: `SELECT COUNT(*) AS order_count FROM orders WHERE o_totalprice BETWEEN 100000 AND 200000`
- in_list: `SELECT COUNT(*) AS line_count FROM lineitem WHERE l_shipmode IN ('MAIL', 'SHIP')`
- not_in_list: `SELECT COUNT(*) AS line_count FROM lineitem WHERE l_shipmode NOT IN ('MAIL', 'SHIP')`
- like: `SELECT COUNT(*) AS part_count FROM part WHERE p_type LIKE '%BRASS'`
- not_like: `SELECT COUNT(*) AS part_count FROM part WHERE p_type NOT LIKE 'PROMO%'`
- case_when: `SELECT o_orderpriority, SUM(CASE WHEN o_orderstatus = 'F' THEN 1 ELSE 0 END) AS finished FROM orders GROUP BY o_orderpriority`
- date_literal: `SELECT COUNT(*) AS order_count FROM orders WHERE o_orderdate >= DATE '1995-01-01' AND o_orderdate < DATE '1996-01-01'`
- date_compare_string: `SELECT COUNT(*) AS order_count FROM orders WHERE o_orderdate >= '1995-01-01' AND o_orderdate < '1996-01-01'`
- arithmetic: `SELECT l_linenumber, l_extendedprice * (1 - l_discount) * (1 + l_tax) AS charge FROM lineitem WHERE l_orderkey = 1 ORDER BY l_linenumber`
- integer_division: `SELECT p_partkey, p_size, p_size / 10 AS size_band FROM part WHERE p_partkey <= 5 ORDER BY p_partkey`

## Doesn't work

- having: not supported by this engine
- order_by_aggregate: not supported by this engine
- order_by_position: runs, but gives a different answer than SQLite
- offset: not supported by this engine
- self_join: a table can appear only once in a query
- left_join: LEFT JOIN runs as an inner join here, so don't use it
- distinct: not supported by this engine
- count_distinct: not supported by this engine
- subquery_where: not supported by this engine
- subquery_from: not supported by this engine
- with_cte: not supported by this engine
- union: not supported by this engine
- is_null: there are no NULLs
- coalesce: not supported by this engine
- extract: not supported by this engine
- substring: not supported by this engine
- upper: not supported by this engine
- cast: not supported by this engine
- select_no_from: not supported by this engine
