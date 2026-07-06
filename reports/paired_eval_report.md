# Paired Eval Report (OKF vs Flat RAG vs Agentic RAG)

- Run timestamp (UTC): `2026-07-06T05:46:51+00:00`
- Eval set: `eval/eval_set_v1.yaml`
- Models: generation `qwen3.5:4b`, embeddings `nomic-embed-text`
- Dataset: `olistbr/brazilian-ecommerce` (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## What Is Compared

- Same 14 questions, run in one execution.
- OKF method uses `bundle/` concepts (Notebook 1-style).
- Flat methods regenerate a notebook-2-style flat corpus and ingest into ChromaDB.

## Limitations

- Small eval set; a single question can swing percentages noticeably.
- Local model variance can change answers/citations between runs.
- This is a hands-on architectural comparison, not a statistically-powered benchmark.

## Aggregate Scorecard

| Method | Retrieval hit rate | Answer citation-hit | Distractor refusal correctness | Avg calls / question | Avg searches / question |
|---|---:|---:|---:|---:|---:|
| OKF RAG (FAISS + hybrid rerank) | 100.0% | 91.7% | 100.0% | 3.00 | 1.00 |
| Flat Simple RAG (ChromaDB) | 100.0% | 91.7% | 100.0% | 2.00 | 1.00 |
| Flat Agentic RAG (tool-calling) | 91.7% | 58.3% | 100.0% | 3.29 | 1.14 |

## Per-Question Side-by-Side Answers

### Q1. What columns does the orders table have?

- Category: `direct`
- Answerable: `True`
- Expected concept id(s): `['tables/orders']`

#### OKF RAG

- Retrieved ids: `['tables/order_reviews', 'tables/order_payments', 'tables/order_items', 'tables/orders', 'references/metrics/top_product_categories', 'references/metrics/avg_order_value', 'tables/products', 'references/metrics/payment_type_distribution']`
- Final context ids: `['tables/order_reviews', 'tables/order_payments', 'tables/orders', 'tables/products']`
- Citations in answer: `['tables/orders']`

```text
The **orders** table contains eight specific columns that track order details such as customer information, status, and various delivery timestamps. These columns include `order_id`, `customer_id`, `order_status`, purchase timestamp (`order_purchase_timestamp`), approval time (`order_approved_at`), carrier delivery date (`order_delivered_carrier_date`), actual customer delivery date (`order_delivered_customer_date`), and estimated delivery date (`order_estimated_delivery_date`) [tables/orders].
```

#### Flat Simple RAG

- Retrieved ids: `['tables/orders', 'tables/order_payments', 'tables/order_reviews', 'tables/order_items']`
- Citations in answer: `[]`

```text
The `orders` table contains eight string fields that track various timestamps and statuses for individual transactions. These specific columns are `order_id`, `customer_id`, `order_status`, `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, and `order_estimated_delivery_date`.
```

#### Flat Agentic RAG

- Retrieved ids: `['tables/orders', 'tables/order_payments', 'tables/order_items']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['tables/orders']`

```text
The **orders** table contains eight string fields that track various timestamps and statuses for individual transactions. Based on the search results, these columns include:

1.  `order_id` (identifier)
2.  `customer_id` (foreign key linking to customers)
3.  Purchase time (temporal data point)
4.  Estimated delivery date (never null)
5.  Actual delivered dates (which have varying rates of missing values ranging from 0.16% to nearly 3%)

The remaining three columns are not explicitly named in the provided text, but they relate to other temporal or status tracking fields for individual transactions [tables/orders].
```

### Q2. What is the average order value?

- Category: `direct`
- Answerable: `True`
- Expected concept id(s): `['references/metrics/avg_order_value']`

#### OKF RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'references/metrics/avg_review_score', 'references/metrics/review_score_distribution', 'references/metrics/late_delivery_rate', 'references/metrics/payment_type_distribution', 'tables/order_reviews', 'references/metrics/top_product_categories', 'tables/order_payments']`
- Final context ids: `['references/metrics/avg_order_value', 'references/metrics/avg_review_score', 'tables/order_payments', 'tables/order_reviews']`
- Citations in answer: `['references/metrics/avg_order_value', 'tables/order_payments']`

```text
The average order value on this platform is exactly 160.99 BRL [references/metrics/avg_order_value]. This metric was calculated by summing all payment values within each of the 99,440 records in the `order_payments` table and then averaging those totals [tables/order_payments].
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'references/metrics/avg_review_score', 'references/metrics/review_score_distribution', 'tables/orders']`
- Citations in answer: `['references/metrics/avg_order_value', 'tables/orders']`

```text
The average order value is R$ 160.99 BRL [references/metrics/avg_order_value]. This figure represents the mean sum of payment values calculated across a dataset containing 99,440 records from the `order_payments` table [tables/orders][references/metrics/avg_order_value].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'tables/orders', 'references/metrics/review_score_distribution']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['references/metrics/avg_order_value']`

```text
The average order value is R$ 160.99 [references/metrics/avg_order_value]. This metric represents the mean sum of payment values across a dataset containing 99,440 records and serves as a standardized benchmark for revenue analysis within your business operations during this reporting period.
```

### Q3. What is the distribution of payment types?

- Category: `direct`
- Answerable: `True`
- Expected concept id(s): `['references/metrics/payment_type_distribution']`

#### OKF RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'references/metrics/review_score_distribution', 'references/metrics/top_product_categories', 'tables/sellers', 'references/metrics/late_delivery_rate', 'tables/customers']`
- Final context ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'tables/sellers']`
- Citations in answer: `['references/metrics/payment_type_distribution', 'tables/order_payments']`

```text
The payment type distribution shows that credit cards dominate usage at 73.92%, followed by boleto payments at 19.04% [references/metrics/payment_type_distribution]. The remaining segments consist of vouchers at 5.56%, debit card transactions at 1.47%, and no undefined entries recorded in the dataset [tables/order_payments].
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'tables/sellers']`
- Citations in answer: `['references/metrics/payment_type_distribution', 'tables/order_payments']`

```text
The payment type distribution shows that credit cards dominate usage at 73.92%, followed by boleto payments at 19.04% and vouchers at 5.56%. Debit card transactions account for a minimal 1.47% of the total volume, while not_defined entries represent 0.0% [references/metrics/payment_type_distribution]. This analysis was calculated using `value_counts(payment_type, normalize=True)` across 103,886 records from the order_payments table [tables/order_payments][references/metrics/payment_type_distribution].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'tables/sellers', 'references/metrics/top_product_categories']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['tables/orders']`

```text
The distribution of payment types shows a clear dominance by credit cards, which account for 73.92% of all transactions [tables/orders]. Boleto payments follow as the second most common method at 19.04%, while vouchers make up 5.56%. Debit card usage is minimal at just 1.48%, and there are no undefined payment types in this dataset, which contains over 103k records [tables/orders].
```

### Q4. How can product category names be translated to English?

- Category: `direct`
- Answerable: `True`
- Expected concept id(s): `['tables/category_translation']`

#### OKF RAG

- Retrieved ids: `['tables/category_translation', 'references/metrics/top_product_categories', 'tables/products', 'datasets/olist_ecommerce', 'references/metrics/avg_review_score', 'references/metrics/review_score_distribution', 'tables/order_items', 'tables/order_reviews']`
- Final context ids: `['tables/category_translation', 'references/metrics/top_product_categories', 'tables/products', 'tables/order_items']`
- Citations in answer: `['tables/category_translation']`

```text
Product category names are translated into English using the **category_translation** table, which maps each original term directly to its corresponding English equivalent. For example, `beleza_saude` translates to `health_beauty`, and `automotivo` corresponds to `auto`. This dataset contains 71 complete mappings with no missing data [tables/category_translation].
```

#### Flat Simple RAG

- Retrieved ids: `['tables/category_translation', 'tables/products', 'references/metrics/top_product_categories', 'datasets/olist_ecommerce']`
- Citations in answer: `['tables/category_translation', 'tables/products']`

```text
Product category names are stored in a bilingual format within the `category_translation` table, which contains exactly two string fields (`product_category_name` and `product_category_name_english`) that hold identical data for every record [tables/category_translation]. This structure acts as an exact mirror of the main products table, ensuring a direct one-to-one correspondence where each category name is available in both Portuguese and English without missing values or inconsistencies [tables/products]. To retrieve the translation, users can join the `products` table with `category_translation` on the matching product category names to access the English version directly from the dataset.
```

#### Flat Agentic RAG

- Retrieved ids: `['tables/category_translation', 'tables/products', 'references/metrics/top_product_categories']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['tables/category_translation']`

```text
Based on the search results, product category names are already available directly in English within your dataset without requiring a separate translation process. The `category_translation` table contains both `product_category_name` and `product_category_name_english`, which store identical data for every record [tables/category_translation]. This means you can simply reference the `product_category_name_english` field to get the translated names directly from your existing product records.
```

### Q5. How much do customers typically spend per basket?

- Category: `paraphrase`
- Answerable: `True`
- Expected concept id(s): `['references/metrics/avg_order_value']`

#### OKF RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'references/metrics/payment_type_distribution', 'references/metrics/top_product_categories', 'tables/customers', 'datasets/olist_ecommerce', 'references/metrics/review_score_distribution', 'tables/orders', 'tables/order_payments']`
- Final context ids: `['references/metrics/avg_order_value', 'references/metrics/payment_type_distribution', 'tables/order_payments', 'tables/orders']`
- Citations in answer: `['references/metrics/avg_order_value', 'tables/order_payments', 'tables/orders']`

```text
Customers on the platform typically spend an average of exactly 160.99 BRL per completed purchase [references/metrics/avg_order_value]. This figure is derived by summing all payment values within each order and averaging those totals across nearly 100,000 records in the dataset [tables/order_payments]. The metric reflects the mean transaction amount rather than a single basket size, as orders may contain multiple items or payments [references/metrics/avg_order_value][tables/orders].
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'references/metrics/avg_order_value', 'tables/customers', 'datasets/olist_ecommerce']`
- Citations in answer: `['tables/orders', 'references/metrics/avg_order_value']`

```text
Customers typically spend an average of R$ 160.99 per order based on the mean sum of payment values grouped by individual orders [tables/orders]. This figure represents a standardized benchmark for transaction size derived from 99,440 records within the dataset [references/metrics/avg_order_value].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'references/metrics/payment_type_distribution', 'tables/order_payments']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['references/metrics/avg_order_value']`

```text
Customers typically spend an average of R$ 160.99 per basket, based on the mean sum of payment values across over 99,440 recorded orders [references/metrics/avg_order_value]. This figure serves as a key benchmark for understanding typical transaction sizes within your business operations during this reporting period.
```

### Q6. What fraction of shipments arrive behind schedule?

- Category: `paraphrase`
- Answerable: `True`
- Expected concept id(s): `['references/metrics/late_delivery_rate']`

#### OKF RAG

- Retrieved ids: `['references/metrics/late_delivery_rate', 'references/metrics/top_product_categories', 'references/metrics/payment_type_distribution', 'references/metrics/avg_order_value', 'tables/order_items', 'tables/orders', 'references/metrics/review_score_distribution', 'tables/order_reviews']`
- Final context ids: `['references/metrics/late_delivery_rate', 'references/metrics/top_product_categories', 'tables/orders', 'references/metrics/review_score_distribution']`
- Citations in answer: `['references/metrics/late_delivery_rate', 'tables/orders']`

```text
Approximately 8.11% of orders were delivered after their estimated date [references/metrics/late_delivery_rate]. This metric is calculated by comparing the actual customer delivery timestamp against the estimated delivery date for over ninety-six thousand completed shipments [tables/orders]. Consequently, roughly eight out of every one hundred records in the system experienced a delay beyond the promised timeframe.
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/late_delivery_rate', 'tables/orders', 'references/metrics/avg_order_value', 'references/metrics/payment_type_distribution']`
- Citations in answer: `['references/metrics/late_delivery_rate']`

```text
Based on the provided knowledge base, exactly **8.11%** of delivered orders arrived later than their originally estimated dates [references/metrics/late_delivery_rate]. This metric is calculated by measuring the proportion of shipments where the actual delivery date exceeded the estimate across a dataset containing 96,470 records [references/metrics/late_delivery_rate], indicating that over eight percent of fulfilled transactions were behind schedule.
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/late_delivery_rate', 'tables/orders', 'references/metrics/avg_order_value', 'references/metrics/payment_type_distribution', 'references/metrics/top_product_categories']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['references/metrics/late_delivery_rate']`

```text
Based on the search results, **8.11%** of shipments arrive behind schedule [references/metrics/late_delivery_rate]. This metric represents the proportion of delivered orders where the actual delivery date exceeded the originally estimated date, calculated across a dataset containing 96,470 records from the `orders` table.
```

### Q7. Are Brazilian customers happy with their purchases based on ratings?

- Category: `paraphrase`
- Answerable: `True`
- Expected concept id(s): `['references/metrics/avg_review_score', 'references/metrics/review_score_distribution']`

#### OKF RAG

- Retrieved ids: `['datasets/olist_ecommerce', 'references/metrics/review_score_distribution', 'references/metrics/top_product_categories', 'references/metrics/avg_review_score', 'tables/order_reviews', 'tables/customers', 'tables/sellers', 'references/metrics/payment_type_distribution']`
- Final context ids: `['datasets/olist_ecommerce', 'references/metrics/review_score_distribution', 'references/metrics/avg_review_score', 'tables/order_reviews']`
- Citations in answer: `['references/metrics/avg_review_score', 'references/metrics/review_score_distribution']`

```text
Yes, Brazilian customers are generally highly pleased with their purchases, as evidenced by an average review score of exactly 4.09 stars out of a possible five [references/metrics/avg_review_score]. Furthermore, the vast majority of reviews express high satisfaction, with over half (57.78%) giving products a perfect five-star rating [references/metrics/review_score_distribution]. Although small fractions rated services as poor or average, these negative sentiments represent only minor portions compared to the overwhelmingly positive feedback received by sellers and customers alike [references/metrics/review_score_distribution].
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/review_score_distribution', 'datasets/olist_ecommerce', 'references/metrics/avg_review_score', 'tables/customers']`
- Citations in answer: `['references/metrics/review_score_distribution', 'references/metrics/avg_review_score']`

```text
Yes, Brazilian customers appear highly satisfied with their purchases as the review score distribution shows a vast majority of feedback is positive, with 57.78% receiving perfect five-star ratings [references/metrics/review_score_distribution]. This strong overall customer satisfaction level and consistent service quality are further supported by an average review score of exactly 4.09 stars out of a possible five across the dataset [references/metrics/avg_review_score].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/review_score_distribution', 'datasets/olist_ecommerce', 'tables/order_reviews', 'references/metrics/avg_review_score', 'tables/customers']`
- Searches: `2`; Iterations: `3`; Capped: `False`
- Citations in answer: `['1', '2']`

```text
Based on the available data from the Brazilian e-commerce platform, customers appear to be quite satisfied overall. The average review score is **4.09 out of 5 stars**, and the distribution shows that a strong majority (over half) leave perfect five-star ratings [1][2]. While there are some lower scores present—such as roughly 8% for three or four stars—the concentration of highly positive feedback suggests consistent service quality across the platform's customer base.
```

### Q8. If I know a product's category in Portuguese, how do I find its seller's city?

- Category: `multi-hop`
- Answerable: `True`
- Expected concept id(s): `['tables/category_translation', 'tables/order_items', 'tables/sellers']`

#### OKF RAG

- Retrieved ids: `['tables/category_translation', 'tables/geolocation', 'datasets/olist_ecommerce', 'tables/sellers', 'tables/customers', 'references/metrics/top_product_categories', 'tables/products', 'tables/order_items']`
- Final context ids: `['tables/category_translation', 'tables/geolocation', 'datasets/olist_ecommerce', 'tables/products']`
- Citations in answer: `['tables/category_translation', 'tables/products', 'datasets/olist_ecommerce']`

```text
To find the seller's city from a known Portuguese product category, you must first join the **category_translation** table to convert the term into English and then link it through the **products** table using `product_category_name` [tables/category_translation], followed by joining with **order_items** via `product_id` to access associated orders. However, based on the provided schema, there is no direct path from a product category or seller ID to a specific city because the **sellers** and **geolocation** tables are not shown as joined in this context [tables/products], [datasets/olist_ecommerce]. The dataset does contain geocoding data for cities via zip codes, but without explicit links between sellers' locations and their product categories in the available join information.
```

#### Flat Simple RAG

- Retrieved ids: `['tables/sellers', 'datasets/olist_ecommerce', 'tables/products', 'tables/customers']`
- Citations in answer: `['tables/products', 'tables/sellers']`

```text
To find the seller's city given a product's category in Portuguese, you must first join the **products** table to the **category_translation** table using `product_category_name` to retrieve the English category name [tables/products]. Next, link this result to the **sellers** table via the common foreign key relationship between order items and sellers (implied through product-to-order-item linkage) or by matching specific geographic attributes if direct seller-product links exist in your query logic; however, the provided schema does not show a direct join path from `products` to `sellers`, requiring an intermediate step through **order_items** which connects both entities [tables/products][tables/sellers]. Finally, once connected via order transactions, you can access the `seller_city` column from the sellers table using their unique identifier [tables/sellers].
```

#### Flat Agentic RAG

- Retrieved ids: `['tables/sellers', 'tables/geolocation', 'tables/products', 'tables/customers', 'references/metrics/top_product_categories', 'tables/order_items', 'tables/orders']`
- Searches: `2`; Iterations: `3`; Capped: `False`
- Citations in answer: `['tables/products', 'tables/order_items', 'tables/sellers']`

```text
To find the seller's city based on their product category, you would need to join three tables: **products**, **order_items**, and **sellers**. The process involves starting with your known Portuguese product category (e.g., "eletrônicos"), finding all products that match this category in the `products` table using a string comparison or SQL LIKE operator. Next, use those product IDs from the `order_items` table to identify which sellers sold them via their corresponding seller ID. Finally, look up each unique seller's city name by joining with the geographic attributes stored directly within the `sellers` table [tables/products][tables/order_items][tables/sellers].
```

### Q9. How would I compute total revenue per seller?

- Category: `multi-hop`
- Answerable: `True`
- Expected concept id(s): `['tables/order_items', 'tables/order_payments', 'tables/sellers']`

#### OKF RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'references/metrics/review_score_distribution', 'datasets/olist_ecommerce', 'references/metrics/top_product_categories', 'tables/sellers', 'references/metrics/avg_review_score', 'references/metrics/payment_type_distribution', 'tables/orders']`
- Final context ids: `['references/metrics/avg_order_value', 'references/metrics/review_score_distribution', 'datasets/olist_ecommerce', 'tables/orders']`
- Citations in answer: `['sellers', 'orders', 'tables/orders', 'references/metrics/avg_order_value']`

```text
To compute the total revenue per seller, you must first join the `[sellers]` and `[orders]` tables on their respective IDs to link each order to its corresponding seller. Next, aggregate all payment values from the `order_payments` table by grouping them with these joined records using a function like `sum(payment_value)` grouped by `seller_id`. This approach sums every transaction amount associated with a specific seller across the dataset [tables/orders][references/metrics/avg_order_value].
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'references/metrics/payment_type_distribution', 'references/metrics/review_score_distribution', 'tables/sellers']`
- Citations in answer: `['references/metrics/avg_order_value', 'tables/sellers']`

```text
To compute the total revenue per seller, you should sum the payment values from the `order_payments` source table and group these sums by the corresponding `seller_id`. This calculation requires joining the order payments data with the sellers dimension to link each transaction's monetary value to its specific merchant record. The resulting metric would represent the aggregate income generated by individual sellers within your dataset [references/metrics/avg_order_value][tables/sellers].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/avg_order_value', 'tables/sellers', 'datasets/olist_ecommerce']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `[]`

```text
To compute total revenue per seller in the Olist dataset, you would typically aggregate payment values associated with orders linked to each specific `seller_id`. Based on the available tables and metrics, this involves joining data from the **sellers** table (which identifies unique sellers via their primary key) with transactional records like **order_payments**, summing up all corresponding payment amounts for each seller.

The general SQL approach would look something like:
```sql
SELECT s.seller_id, SUM(op.payment_value) AS total_revenue
FROM sellers s
JOIN order_items oi ON s.seller_id = oi.seller_id
JOIN orders o ON oi.order_id = o.order_id
JOIN order_payments op ON o.order_id = op.order_id
GROUP BY s.seller_id;
```

This calculation sums the `payment_value` across all payments for each seller, effectively giving you their total revenue.
```

### Q10. Which table tells me both a customer's zip code and their approximate map coordinates?

- Category: `multi-hop`
- Answerable: `True`
- Expected concept id(s): `['tables/customers', 'tables/geolocation']`

#### OKF RAG

- Retrieved ids: `['tables/geolocation', 'tables/customers', 'tables/sellers', 'tables/orders', 'tables/order_items', 'tables/order_payments', 'tables/products', 'datasets/olist_ecommerce']`
- Final context ids: `['tables/geolocation', 'tables/customers', 'tables/sellers', 'datasets/olist_ecommerce']`
- Citations in answer: `['tables/geolocation', 'tables/customers']`

```text
The `geolocation` table provides the specific latitude and longitude coordinates for geographic locations defined by zip code prefixes [tables/geolocation]. While this table contains the coordinate data linked to a prefix that can be matched to customers via join relationships, it does not directly store customer-specific identifiers like their unique ID or full name. To obtain both a customer's identity (such as `customer_id`) and their approximate map coordinates in a single view, you would need to perform a join between the `customers` table—which holds the zip code prefix—and the `geolocation` table [tables/customers][tables/geolocation].
```

#### Flat Simple RAG

- Retrieved ids: `['tables/geolocation', 'tables/customers', 'tables/sellers', 'tables/orders']`
- Citations in answer: `['tables/geolocation', 'tables/customers']`

```text
The **geolocation** table provides the specific latitude and longitude coordinates for geographic locations [tables/geolocation]. While this table contains the coordinate data itself, it does not directly store customer identifiers; instead, you must join it with the **customers** table using `customer_zip_code_prefix` to link a zip code prefix (which represents the approximate location) to an individual customer record [tables/customers][tables/geolocation].
```

#### Flat Agentic RAG

- Retrieved ids: `['tables/geolocation', 'tables/customers', 'tables/sellers']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['tables/geolocation']`

```text
Based on the search results, the **`geolocation`** table contains both zip code and map coordinates for customers. It includes five fields: an integer `zip_code_prefix`, latitude and longitude as floats, along with city and state identifiers [tables/geolocation]. While the customer records themselves are in the `customers` table which only stores a numeric `customer_zip_code_prefix` without explicit coordinate data, every record in the `geolocation` table aligns perfectly (100% overlap) with these customer zip code prefixes.
```

### Q11. seller performance

- Category: `vague`
- Answerable: `True`
- Expected concept id(s): `['tables/order_items', 'tables/sellers']`

#### OKF RAG

- Retrieved ids: `['tables/sellers', 'datasets/olist_ecommerce', 'references/metrics/late_delivery_rate', 'references/metrics/review_score_distribution', 'references/metrics/avg_review_score', 'tables/order_reviews', 'tables/orders', 'tables/order_items']`
- Final context ids: `['tables/sellers', 'datasets/olist_ecommerce', 'tables/orders', 'tables/order_items']`
- Citations in answer: `['datasets/olist_ecommerce', 'tables/sellers', 'tables/order_items']`

```text
Seller performance in the Olist dataset is tracked through three thousand unique sellers who participated in transactions between September 2016 and October 2018 [datasets/olist_ecommerce]. Each seller's geographic location can be identified by their zip code prefix, city, state, or specific ID linked to order items [tables/sellers][tables/order_items]. While the dataset provides comprehensive records of individual line item prices and freight values for these sellers, it does not explicitly define a "performance" metric such as sales volume rankings or profitability ratios within the provided schema.
```

#### Flat Simple RAG

- Retrieved ids: `['tables/sellers', 'references/metrics/late_delivery_rate', 'references/metrics/review_score_distribution', 'tables/order_reviews']`
- Citations in answer: `['references/metrics/late_delivery_rate', 'references/metrics/review_score_distribution', 'tables/sellers', 'tables/order_reviews']`

```text
Seller performance is primarily reflected in the late delivery rate of 8.11%, which indicates that over eight percent of fulfilled orders arrived later than their estimated dates [references/metrics/late_delivery_rate]. Additionally, customer satisfaction appears strong based on review data showing a majority score distribution with 57.78% receiving perfect five-star ratings out of nearly 100,000 reviews [references/metrics/review_score_distribution]. The sellers table itself provides geographic context but does not contain direct performance metrics like delivery times or scores; those are derived from the orders and order_reviews tables respectively [tables/sellers], [tables/order_reviews].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/avg_review_score', 'references/metrics/late_delivery_rate', 'references/metrics/review_score_distribution', 'references/metrics/avg_order_value', 'references/metrics/top_product_categories']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['references/metrics/avg_review_score', 'references/metrics/late_delivery_rate', 'references/metrics/top_product_categories']`

```text
Based on the search results, seller performance is evaluated using several key metrics including customer satisfaction (measured by an average review score of 4.09 stars), late delivery rates (currently at 8.11%), and order value trends [references/metrics/avg_review_score][references/metrics/late_delivery_rate]. Additionally, the top product categories driving transaction volume include bed_bath_table, health_beauty, and sports_leisure items [references/metrics/top_product_categories], which can serve as indicators of seller performance in terms of sales velocity.
```

### Q12. payment methods

- Category: `vague`
- Answerable: `True`
- Expected concept id(s): `['references/metrics/payment_type_distribution', 'tables/order_payments']`

#### OKF RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'tables/sellers', 'references/metrics/late_delivery_rate', 'tables/orders', 'datasets/olist_ecommerce', 'tables/customers']`
- Final context ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'tables/orders', 'references/metrics/avg_order_value']`
- Citations in answer: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'tables/orders']`

```text
Based on the provided knowledge base, credit cards are the dominant payment method, accounting for **73.92%** of transactions [references/metrics/payment_type_distribution]. This is followed significantly by boleto payments at 19.04%, while debit card usage represents only a small segment with 1.47% and undefined entries account for none of the records [tables/order_payments]. The data indicates that these payment types are recorded in the `order_payments` table, which contains no null values across all fields [tables/orders].
```

#### Flat Simple RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'tables/sellers']`
- Citations in answer: `['references/metrics/payment_type_distribution', 'tables/order_payments']`

```text
The dataset identifies five distinct payment types based on the distribution metric [references/metrics/payment_type_distribution], where credit cards dominate usage at 73.92%, followed by boleto payments at 19.04% and vouchers at 5.56%. Debit card transactions account for a minimal 1.47% of total volume, while not_defined entries represent 0.0% of the data [references/metrics/payment_type_distribution]. These payment records are stored in the `order_payments` table, which contains complete information on all five columns including payment type without any null values [tables/order_payments].
```

#### Flat Agentic RAG

- Retrieved ids: `['references/metrics/payment_type_distribution', 'tables/order_payments', 'references/metrics/avg_order_value', 'tables/sellers', 'references/metrics/late_delivery_rate']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['1', '2']`

```text
Based on the data retrieved from your knowledge base, **credit cards** are by far the most popular payment method, accounting for approximately **73.92%** of all transactions [1]. The second and third most common methods are **boleto payments**, which make up about 19.04% of sales, followed closely by **vouchers** at roughly 5.56%. Debit card usage is minimal, representing only 1.47% of the total volume [2].

These payment types were identified through a distribution analysis across over 103,886 transaction records in your dataset [1][2].
```

### Q13. What is the customer's email address or phone number?

- Category: `distractor`
- Answerable: `False`
- Expected concept id(s): `[]`

#### OKF RAG

- Retrieved ids: `['tables/customers', 'tables/orders', 'tables/geolocation', 'datasets/olist_ecommerce', 'tables/sellers', 'references/metrics/top_product_categories', 'references/metrics/avg_order_value', 'tables/order_items']`
- Final context ids: `['tables/customers', 'tables/orders', 'tables/geolocation', 'tables/order_items']`
- Citations in answer: `['tables/customers', 'tables/orders']`

```text
Based on the provided knowledge base entries, there is no information regarding customers' email addresses or phone numbers. The schema for the `customers` table includes only identifiers (`customer_id`, `customer_unique_id`) and location details such as city and state [tables/customers]. Similarly, other tables like `orders` do not contain personal contact information beyond timestamps and status data [tables/orders].
```

#### Flat Simple RAG

- Retrieved ids: `['tables/customers', 'tables/sellers', 'tables/geolocation', 'tables/orders']`
- Citations in answer: `['tables/customers']`

```text
The provided knowledge base entries do not contain information regarding customers' email addresses or phone numbers. The attributes listed for the `customers` table are limited to an alphanumeric ID, a numeric zip code prefix, and location fields such as city and state [tables/customers]. Consequently, based strictly on these documents, it is impossible to determine any contact details like emails or phone numbers for the customers in this dataset.
```

#### Flat Agentic RAG

- Retrieved ids: `['tables/customers', 'tables/sellers', 'tables/geolocation', 'datasets/olist_ecommerce', 'tables/orders']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['tables/customers', 'tables/orders', 'tables/geolocation']`

```text
Based on the search results, this dataset does not contain customer email addresses or phone numbers. The available tables include:

- **customers**: Contains alphanumeric IDs (`customer_id`, `customer_unique_id`), zip code prefix, city, and state [tables/customers]
- **orders**: Contains order timestamps, statuses, and links to customers via `customer_id` [tables/orders]
- **geolocation**: Contains geographic coordinates (latitude/longitude) linked by zip code prefix [tables/geolocation]

The customer contact information is limited to location data only. If you need specific email or phone numbers for a particular transaction, they would not be present in this dataset's structure.
```

### Q14. What was Olist's total company revenue in 2020?

- Category: `distractor`
- Answerable: `False`
- Expected concept id(s): `[]`

#### OKF RAG

- Retrieved ids: `['datasets/olist_ecommerce', 'references/metrics/top_product_categories', 'references/metrics/avg_order_value', 'references/metrics/review_score_distribution', 'references/metrics/late_delivery_rate', 'references/metrics/avg_review_score', 'references/metrics/payment_type_distribution', 'tables/customers']`
- Final context ids: `['datasets/olist_ecommerce', 'references/metrics/top_product_categories', 'references/metrics/avg_order_value', 'references/metrics/payment_type_distribution']`
- Citations in answer: `['datasets/olist_ecommerce']`

```text
The provided knowledge base entries do not contain information about Olist's total company revenue for the year 2020. The dataset covers e-commerce transactions from September 2016 to October 2018 and includes metrics on order values, payment types, and product categories, but it does not report corporate financial statements or annual revenue figures [datasets/olist_ecommerce].
```

#### Flat Simple RAG

- Retrieved ids: `['datasets/olist_ecommerce', 'references/metrics/avg_order_value', 'references/metrics/late_delivery_rate', 'references/metrics/top_product_categories']`
- Citations in answer: `['datasets/olist_ecommerce', 'references/metrics/avg_order_value']`

```text
The provided knowledge base does not contain information about Olist's total company revenue for the year 2020 because the dataset only spans from September 4, 2016, to October 17, 2018 [datasets/olist_ecommerce]. Additionally, while an average order value of R$ 160.99 is reported based on approximately 99,440 records in the dataset [references/metrics/avg_order_value], there are no specific figures or calculations regarding total revenue for any year after October 2018.
```

#### Flat Agentic RAG

- Retrieved ids: `['datasets/olist_ecommerce', 'references/metrics/avg_order_value', 'references/metrics/avg_review_score']`
- Searches: `1`; Iterations: `2`; Capped: `False`
- Citations in answer: `['datasets/olist_ecommerce']`

```text
The search results do not contain information about Olist's total company revenue in 2020. The available documents describe a dataset covering the period from September 2016 to October 2018, which predates 2020 and does not include financial metrics like annual revenue for that year [datasets/olist_ecommerce].
```
