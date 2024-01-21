import pandas as pd
import numpy as np

class Moons():
    
    def __init__(self):

        import pandas as pd
        import numpy as np

        database_service = "sqlite"
        database = "data/jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "moons"
        moon_df = pd.read_sql(query, connectable)
        
        self.data = moon_df
        self.name = moon_df["moon"]

    
    def moon_info(self, moon_name, attribute = None):
        moon_row = self.data[self.data["moon"] == moon_name]
        info = {
        "period" : moon_row["period_days"].iloc[0],
        "distance" : moon_row["distance_km"].iloc[0],
        "radius" : moon_row["radius_km"].iloc[0],
        "mag" : moon_row["mag"].iloc[0],
        "mass" : moon_row["mass_kg"].iloc[0],
        "group" : moon_row["group"].iloc[0],
        "ecc" : moon_row["ecc"].iloc[0],
        "inclination" : moon_row["inclination_deg"].iloc[0]}
        if attribute in info:
            print(f"The {attribute} of '{moon_name}' is {info[attribute]}.")
            
    def regression(self):
        
        T_squared = ((self.data["period_days"]) * 24 * 60 * 60)**2
        a_cubed = ((self.data["distance_km"]) *10**3 )**3
        
        self.data["T_squared"] = T_squared
        self.data["a_cubed"] = a_cubed
        
        proportional = (a_cubed*4*np.pi**2)/ (6.67*(10**-11))
        
        self.data["proportional"] = proportional
        
        X = self.data[["proportional"]]
        Y = self.data["T_squared"]

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 
        
        from sklearn import linear_model
        model = linear_model.LinearRegression(fit_intercept=True)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        
        import seaborn as sns
        sns.relplot(data=self.data,x="proportional",y="T_squared")
        
        gradient = model.coef_[0]
        mass = 1/gradient
        print(mass)
        from sklearn.metrics import r2_score, mean_squared_error

        print(f"r2_score: {r2_score(y_test, pred)}")

        
moon_instance = Moons()
moon_instance.regression()
search_for_moon = "Ganymede"
moon_info = moon_instance.moon_info(search_for_moon, attribute = "ecc")
