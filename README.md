# Spaceship Titanic - Machine Learning Competition

The goal of this project is to predict whether passengers on the Spaceship Titanic were transported to another dimension, using data from the [Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview).

---

### Project Summary

This project uses supervised machine learning to predict the `Transported` status of passengers on the Spaceship Titanic. I developed a classification pipeline that includes data cleaning using methods such as `.fillna()` for missing values and `.map()` for binary encoding. I applied one-hot encoding to categorical variables and engineered a new feature called `TotalSpend`, which aggregates all passenger amenity charges. After exploring the data through visualizations to guide feature selection, I trained a final model using a `RandomForestClassifier` to generate predictions.

---

### How to Run

1. Clone this repo  
2. Download `train.csv` and `test.csv` from Kaggle and save to `/data/`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the model and generate your submission.csv
   
The predictions will be saved in the root directory as `submission.csv`.

---

### Key Features Used

The model incorporates a range of engineered and encoded features. These include `Age`, binary-encoded indicators such as `CryoSleep` and `VIP`, and one-hot encoded categorical variables like `HomePlanet` and `Destination`. Spending variables across amenities—`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck`—were also included. Additionally, a new feature called `TotalSpend` was created by summing all individual spending fields to capture overall expenditure behavior.


### Visualizations 

The `titanic_notebook.ipynb` file includes several exploratory visualizations that guided feature selection and model design. These visualizations include histograms of `Age` segmented by `Transported` status, feature importance rankings from the Random Forest model, and various boxplots and barplots grouped by outcome to reveal relationships between categorical and numeric features.
