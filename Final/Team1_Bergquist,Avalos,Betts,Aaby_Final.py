from shiny import App, ui, render
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as skm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from shinywidgets import output_widget, render_altair, render_widget
import altair as alt
alt.data_transformers.enable("vegafusion")
 
 
# random forest code
 
data = pd.read_csv('ReducedDataset.csv')
#data = pd.read_csv('Merged_Data.csv') # delete this when u run it  
data['Years of SPD Service'] = data['Years of SPD Service'].replace(['< 1', '<1'], 0.5).astype(float)
data_filtered = data[['Officer Disciplined?', 'Years of SPD Service', 'Subject Race', 'Subject Gender', 'Subject Age', 'Fatal', 'Disposition']].dropna()
X = data_filtered[['Officer Disciplined?', 'Years of SPD Service', 'Subject Race', 'Subject Gender', 'Subject Age','Disposition']]
y = data_filtered['Fatal'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_features = ['Officer Disciplined?', 'Years of SPD Service', 'Subject Race', 'Subject Gender', 'Subject Age', 'Disposition']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
feature_importances = pipeline.named_steps['classifier'].feature_importances_
encoded_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
 
#logistic regression
reduced_n = data.copy()
 
reduced_n['Fatal'] = reduced_n['Fatal'].map({'Yes': 1, 'No': 0})
reduced_n = reduced_n.dropna(subset=['Fatal'])
reduced_n['Fatal'] = reduced_n['Fatal'].astype(int)
features = ['Years of SPD Service', 'Subject Race', 'Officer Disciplined?', 'Disposition', 'Subject Age']
X = reduced_n[features]
y = reduced_n['Fatal']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X  [['Subject Race', 'Officer Disciplined?', 'Disposition']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Subject Race', 'Officer Disciplined?', 'Disposition']))
X = pd.concat([X, encoded_df], axis=1)
X = X.drop(['Subject Race', 'Officer Disciplined?', 'Disposition'], axis=1)
X['Years of SPD Service'] = pd.to_numeric(X['Years of SPD Service'], errors='coerce')  # Convert to numeric
X['Years of SPD Service'] = X['Years of SPD Service'].fillna(X['Years of SPD Service'].median())
 
X['Subject Age'] = pd.to_numeric(X['Subject Age'], errors='coerce')  # Convert to numeric
X['Subject Age'] = X['Subject Age'].fillna(X['Subject Age'].median())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
 
#pca and clustering
#n_data =pd.read_csv('ReducedDataset.csv')
n_data = pd.read_csv('ReducedDataset.csv') #delete
 
 
features = ['Years of SPD Service', 'Subject Race', 'Officer Disciplined?', 'Disposition', 'Subject Age']
data = n_data[features].dropna()
 
for col in features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
pca = PCA(n_components=3)
pca_components = pca.fit_transform(scaled_data)
 
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_components)
n_data['Cluster'] = clusters
 
n_data['Fatal'] = n_data['Fatal'].map({'Yes': 1, 'No': 0})
fatality_rates = n_data.groupby('Cluster')['Fatal'].mean()
 
cluster_names = {
    0: "Cluster with Highest Fatality",
    1: "Cluster with Middle Fatality",
    2: "Cluster with Low Fatality",
}
fatality_rates_percentage = (fatality_rates * 100).round(2)
fatality_rates_named = fatality_rates_percentage.rename(index=cluster_names)
 
 
 
## shiny app ui
app_ui = ui.page_fluid(
    ui.h1('Team 1 - Seattle Crime Data'),
    ui.p('Kikzely Avalos, Mariah Berqguist, Matti Betts, David Aaby'),
    ui.navset_tab(
        ui.nav_panel(
            "Project Overview",
            ui.h2("Overview of the Project"),
            ui.h5('Project Questions: What features are most common in fatal shootings? How can we predict whether a crime will result in a fatality?'),
            ui.p('This project focuses on analyzing fatal and non-fatal shootings in the Seattle area, focusing on the following questions to guide our analysis: What features are most common in fatal shootings? and How can we predict whether a crime will result in a fatality? These questions are important for understanding patterns in law enforcement-related incidents and developing applicable results to reduce fatal crime outcomes. To answer these questions, we analyzed data from two primary datasets. The Seattle SPD Crime Data dataset provided a comprehensive overview of crime rates, incident details, and contextual information for different types of crimes. The SPD Officer-Involved Shooting (OIS) Data offered specific data regarding shootings involving officers, including details on subject demographics, officer characteristics, and the disposition of each incident. We merged the two datasets on the date of crime, and by doing so it allowed us to link general crime information with detailed officer-involved shooting data, allowing for a more detailed dataset for analysis'),
            ui.p('We used the following methods to find patterns in the data and build predictive models that would provide results. Principal Component Analysis (PCA) reduced the dimensionality of the data and highlighted the most influential variables. Clustering techniques were used to reveal groupings in the data, identifying unique characteristics of fatal and non-fatal incidents. For predictive modeling,we used Random Forest to identify the most significant variables, and building off this, we used Logistic Regression to provide interpretable information into how the significant variables provided from the models contribute to fatal outcomes. The “Results” tab provides visualizations regarding our data and results. '),
            ui.p('From the analysis, we saw that policy-related variables (whether incidents were classified as "Within Policy"), subject demographics (age, race, and gender), and officer-related factors (years of service and disciplinary history) were the most significant predictors of fatality in an incident in Seattle. By merging the two datasets, we were able to combine broader crime trends with detailed officer-involved shooting information. These results directly addressed the research questions by identifying the most common features of fatal shootings and providing a predictive framework for understanding these incidents. This project contributes results that can inform policy, training, and interventions to reduce the fatal outcomes in law enforcement related incidents.'),
            ui.p('Links to datasets used: ',
                 ui.a('Seattle Crime Dataset', href='https://data.seattle.gov/Public-Safety/SPD-Crime-Data-2008-Present/tazs-3rd5/about_data', target="_blank"),
                 ' and ',
                 ui.a("SPD Officer-Involved Shootings Dataset", href='https://data.seattle.gov/Public-Safety/SPD-Officer-Involved-Shooting-OIS-Data/mg5r-efcm/about_data', target="_blank")
                 ),
        ),
        ui.nav_panel(
            "Tabulation",
            ui.h2("Dataset"),
            ui.p('Interactive data table for the merged dataset'),
            ui.output_data_frame("data_table"),
        ),
        ui.nav_panel(
            "Results",
            ui.h2("Models and Visualizations of Project Outcome"),
            ui.div(
                ui.h4('3D PCA Scatterplot'),
                output_widget('pca_3d_interactive'),
            ),
            ui.div(
                ui.h3("Overall Feature Importance"),
                ui.output_plot("overall_importance_plot"),
                ui.p('The Random Forest model identifies the important variables that help predict whether a crime incident resulted in fatality by a Seattle Police officer.  Features such as Disposition including Within Policy and Out of Policy and subject demographic features like age, race, and gender are important factors that determine the outcome of these incidents. Disposition Within Policy emerges as the most influential feature, which indicates that whether or not following the protocol of policy has a huge effect on fatality outcomes.'),
                ui.p('The most significant variables that have an impact on the outcome of the crime are Disposititon, Subject Demographics, Officer Experience and if the officer was disciplined.'),
                ui.p('Among all variables, "Disposition Within Policy" is the most significant in determining fatal outcomes. Certain age groups of the subject, particularly those in their mid to late 20s (24,25,29) had more signifiance. Age groups of older inviduals had relatively low importance suggesting a focus on the younger demographic. Variables related to the years of service show moderate importance meaning that while there\'s no trend indicating which age group leads to the most fatality, those with higher years of experience tend to have less fatal incidents. Subject demographcs have a signifcant importance with variables like Black or African American and White being the most associated with fatal outcomes.'),
                style="margin-top:20px;"
            ),
            ui.div(
                ui.div(
                    ui.input_select(
                        "var",
                        "Select Feature to Explore:",
                        choices=list(encoded_feature_names),
                        multiple=False
                    ),
                    style="width:48%; display:inline-block; vertical-align:top; padding-right:10px;"
                ),
                ui.div(
                    ui.output_plot("feature_plot"),
                    style="width:48%; display:inline-block; vertical-align:top;"
                ),
            ),
            ui.div(
                ui.h4('Logistic Regression'),
                ui.input_slider("age", "Subject Age:", min=19, max=78, value=30, step=1),
                ui.input_slider("years_of_service", "Years of Service:", min=1, max=29, value=5, step=1),
                ui.output_plot('logistic_regression'),
            ),
            ui.div(
                ui.h4('Demographic Analysis'),
                ui.input_select(
                    "demographic_var",
                    "Select Demographic Variable:",
                    choices=['Subject Race', 'Subject Gender'],
                    multiple=False
                ),
                ui.output_plot('demographic_plot'),
            )
        )
    )
)
 
def server(input, output, session):
    @output
    @render.data_frame
    def data_table():
        data = pd.read_csv('Merged_Data.csv')
        return render.DataTable(data, filters=True)
 
    @output
    @render.plot
    def feature_plot():
        selected_feature = input.var()
        feature_index = list(encoded_feature_names).index(selected_feature)
        importance_value = feature_importances[feature_index]
        plt.figure(figsize=(9, 3))
        plt.barh([selected_feature], [importance_value], color="#007bc2")
        plt.xlabel("Importance Score")
        plt.title("Selected Feature Importance")
        return plt.gcf()
 
 #max(7, num_features * 0.5)
  

    @output
    @render.plot
    def overall_importance_plot():
        num_features = len(feature_importances)
        #fig_height = max(7, num_features * .6) 
        plt.figure(figsize=(8, 5))
        plt.barh(range(num_features), feature_importances, align="center", height=0.6)
    
        plt.yticks(range(num_features), encoded_feature_names, fontsize=8, rotation=0, ha="right")
    
    
        plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
    
        plt.xlabel("Importance Score", fontsize=14)
        plt.title("Overall Feature Importance", fontsize=16)
    
        #plt.tight_layout() 
        return plt.gcf()

 
 
    @output
    @render.plot
    def logistic_regression():
        age = input.age()
        years_of_service = input.years_of_service()
 
 
        new_data = pd.DataFrame({
            "Years of SPD Service": [years_of_service],
            "Subject Age": [age],
            "Subject Race": ["White"],  
            "Officer Disciplined?": ["No"],
            "Disposition": ["Not Justified"]
        })
        encoded_new_data = encoder.transform(new_data[["Subject Race", "Officer Disciplined?", "Disposition"]])
        encoded_new_data_df = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out(["Subject Race", "Officer Disciplined?", "Disposition"]))
        new_data = pd.concat([new_data, encoded_new_data_df], axis=1).drop(["Subject Race", "Officer Disciplined?", "Disposition"], axis=1)
 
        probability = model.predict_proba(new_data)[:, 1]
 
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Not Fatal", "Predicted Fatal"],
            yticklabels=["Actual Not Fatal", "Actual Fatal"]
        )
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title(f"Confusion Matrix - Logistic Regression\nProbability of Fatality: {probability[0]:.2f}", fontsize=14)
        plt.tight_layout()
        return plt.gcf()
 
 
    @output
    @render.plot
    def demographic_plot():
   
        demographic_var = input.demographic_var()
        demographic_counts = data_filtered.groupby([demographic_var, "Fatal"]).size().unstack().fillna(0)
        demographic_counts = (demographic_counts.div(demographic_counts.sum(axis=1), axis=0) * 100)
        demographic_counts.plot(kind="bar", stacked=True, figsize=(6, 6), color=["#007bc2", "#fdbb84"])
        plt.ylabel("Frequency (%)")
        plt.title(f"Frequency of Fatal Shootings by {demographic_var}")
        plt.xticks(rotation=15)
        plt.legend(title="Fatal", labels=["No", "Yes"])
        plt.tight_layout()
        return plt.gcf()
   
    @output
    @render_widget
    def pca_3d_interactive():
        pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2", "PC3"])
        pca_df['Cluster'] = n_data['Cluster']
        pca_df['Subject Age'] = n_data['Subject Age']
 
        fig = px.scatter_3d(
            pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color ='Cluster',
            title = '3D Scatterplot',
            labels={ "PC1": "Subject Race",
                "PC2": "Officer Disciplined",
                "PC3": "Subject Age"
            },
 
            hover_data=['Subject Age'],
 
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            autosize=True,
            height=600,
            width=800,
            legend=dict(title="Cluster"),
        )
        return fig
    
app = App(app_ui, server)