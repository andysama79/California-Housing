from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit as sss

import fetch_data as fd
import segment

# fetch and store data
fd.fetch_housing_data(fd.HOUSING_URL, fd.HOUSING_PATH)

# load the housing data
housing = fd.load_housing_data()

# take test_ratio from user and segment the data into training set and testing set
test_ratio = int(input("Enter test ratio"))
housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = segment.split_train_test_by_id(housing_with_id, test_ratio, "index")  #or
train_set, test_set = train_test_split(housing, test_size=test_ratio, random_state=42)

# Stratified Shuffle Split
split = sss(n_splits=1, test_size=test_ratio, random_state=42)
for train_index, test_index in split.aplit(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# looking at correlation
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Experimenting with Attribute Combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)




