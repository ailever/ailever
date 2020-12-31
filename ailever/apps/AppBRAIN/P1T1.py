import plotly.express as px

class Components(dict):
    def __init__(self):
        self['0,0'] = None
        self.updateR0C0()

    def updateR0C0(self):
        df = px.data.iris()
        self['0,0'] = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
                "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
                "petal_width": "Petal Width", "petal_length": "Petal Length", },
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)



