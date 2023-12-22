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
