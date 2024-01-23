import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Moons():
    
    def __init__(self):
        
        # loading the moon database and storing it as a data attribute
        
        database_service = "sqlite"
        database = "data/jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "moons"
        moon_df = pd.read_sql(query, connectable)
        
        self.data = moon_df
        
        # storing the names of the moons as an attribute and setting them to the index of the data frame in order to extract >
        
        self.name = moon_df["moon"]
        self.data = self.data.set_index("moon")
        
        
    def moon_info_specific(self, moon_name, attribute):
        
        # finding the row for the desired moon and saving each bit of information into a dictionary so that a characteristic >
        
        moon_row = self.data.loc[moon_name]
        info = {
        "period_days" : moon_row["period_days"],
        "distance_km" : moon_row["distance_km"],
        "radius_km" : moon_row["radius_km"],
        "mag" : moon_row["mag"],
        "mass_kg" : moon_row["mass_kg"],
        "group" : moon_row["group"],
        "ecc" : moon_row["ecc"],
        "inclination_deg" : moon_row["inclination_deg"]}
        
        print(f"The {attribute} of {moon_name} is {info[attribute]}.")
        
    def moon_info(self, moon_name):
        
        # returning all the data on each moon if all the information is wanted
        
        moon_info = self.data.loc[moon_name]
        return moon_info
    
    def correlations(self):
        
        # returns a correlation table for all the information in the data frame
        
        return self.data.corr()
    
    def scatterplots(self, X, Y):
        
        # plots a scatter graph of two chosen characteristics
        
        sns.scatterplot(x=X, y=Y, data=self.data)
        plt.title(f" {Y} against {X}")
        plt.show()
        
    def catplots(self, X, Y, K, O):
        # plots catogorical graph to show how the groups of moons varies         
        sns.catplot(data=self.data, x=X, y=Y, kind=K, order = O)
        plt.title
        plt.show()
        
    def stats(self):
        
        # returns a statistical summary of the data set
        
        print(self.data.describe())
        
        
    def regression(self):
        
        # setting up a regressional model to estimate the mass of jupiter
        
        # converting the period into seconds and squaring for T^2 and the distance from jupiter to the moon to m and cubing f>
        
        T_squared = ((self.data["period_days"]) * 24 * 60 * 60)**2
        a_cubed = ((self.data["distance_km"]) *10**3 )**3
        
        # saving them as columns in the data frame
        
        self.data["T_squared"] = T_squared
        self.data["a_cubed"] = a_cubed
        
        # saving the X and Y variables to train and test on
        
        X = self.data[["a_cubed"]]
        Y = self.data["T_squared"]
        
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # using the hyperparameter of fit_intercept = False to force it through the origin as when the period is zero the dis>
        
        from sklearn import linear_model
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        
        # plotting T^2 against a^3 to show that a linear fit is suitable with a line using the predicted values to validate t>
        
        import seaborn as sns
        sns.relplot(data=self.data,x="a_cubed",y="T_squared")
        plt.plot(x_test, pred, color='cyan', linewidth = 0.5)
        plt.xlabel('Distance (a)^3 / m^3')
        plt.ylabel('Period^2 / s^2')
        plt.show()
        
        # finding the gradient of the graph and calculating the mass of jupiter from it
        
        gradient = model.coef_[0]
        mass = (4*(np.pi**2))/((6.67*(10**-11)*gradient))
        print(f"The approximated mass of Jupiter is {mass}")
        
        # finding  r^2 score to show how closely the model agrees with the data
        from sklearn.metrics import r2_score, mean_squared_error
        
        print(f"The r2_score is {r2_score(y_test, pred)}")
        

