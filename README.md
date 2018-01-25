# RegionSearch
Optimizes pricing regions for ease of price change on products.

#Inspiration

I was given this project from the CEO of Clear Demand, a price analysis company
that has national clients, to assist with controlling their prices in a way so
that they did not have to keep track of every price for each product for each
area separately.

# Data Exploration
The data started out as a CSV file and had 255 columns and 720 rows, about 39 of
which had 1/2 or less rows with values in them. I explored unique values in each
column to determine how much signal I could get in a model and also dropped any
column that had more less than 2 unique values in them. The data had 2 categories
of product, discontinued, and not discontinued, so I made sure to only have current
items in my working data set. I also found that they are controlling 21 separate
pricing combinations across all their areas.

*X values = number of regions -2*

![5_product_rev_box](https://user-images.githubusercontent.com/26101047/35399675-09f9e77c-01b2-11e8-8bf2-1587778c46cd.jpg)


![5_product_profit_box](https://user-images.githubusercontent.com/26101047/35399659-00ff5cec-01b2-11e8-997e-fa56f19fce9b.jpg)


![5_product_margin_box](https://user-images.githubusercontent.com/26101047/35399690-14e71bf0-01b2-11e8-87db-ac3d341bc30f.jpg)


#Clustering Algorithm and Visualization:
 I used KMeans clustering to find correlation areas to other areas using their
 current product pricing. In order to take into account for variance, I ran the
 model 100 times and took the most commonly occurring set of regions. This set
 appeared 32% of the time with second place being 12%.

 ![screen shot 2018-01-25 at 10 16 37 am](https://user-images.githubusercontent.com/26101047/35402161-ed15b6ac-01b8-11e8-9884-da100e491563.png)

#Python Libraries Used
Sklearn
Matplotlib
Seaborn
Pandas
Numpy

![5_product_inertia](https://user-images.githubusercontent.com/26101047/35399602-d7b0d3c0-01b1-11e8-8a61-f8ffbc1cfb94.jpg)
