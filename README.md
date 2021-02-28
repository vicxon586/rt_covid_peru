# Rt - Effective Reproduction Number (rt_covid_peru)
This is the process to get the Effective Reproduction Number (also known as Rt) based on the initial model of Kevin Systrom (http://systrom.com/blog/the-metric-we-need-to-manage-covid-19/) and adapt to work with the data of Peru. This value helps us to understand the contagious rate of the Covid-19 Pandemic. You can also see the article (https://medium.com/@vicxon586/latin-covid-an%C3%A1lisis-de-sudam%C3%A9rica-a-nivel-contagio-b8c5d4ac612c) I wrote based on the construction and adaptation of this model.

In addition, this model is effectively used in the platform OpenCovid-Peru (https://opencovid-peru.com/) we have built with other researchers. We have this report and many others to help the country informing properly about the pandemic.

## 👉🏻 Get Started
- I didn't built a virtual environment for this project, but you can (and should) do it to run the project.
- The format of the input file that you should consider must have the following format:
  - field REGION
  - field DATE (Format YYYY-MM-DD)
  - field of cumulative positive cases (cum_pos_total)
![image](https://user-images.githubusercontent.com/44335731/109433895-f8f8ff80-7a12-11eb-90e9-6cecb83f5242.png)


## 💻 Execute the Process
- First you need to install the packages related (better to build a virtual environment and keep the versions). I will do this in a future version.
```
pip install pandas
pip install numpy
pip install matplotlib
pip install scipy
pip install datetime
```

- Then you need to fill up the variables in the file
```
# For the path you can use the directory of your poject with cwd or define manually a path
path = os.getcwd()
path = '/Users/vicxon586/Documents/MyProjects/Opencovid-Peru/Rt_diarioPeru/'
name_file = '20210224'
```

- Run the process and get the graphs of the Rt Value as well as the values in each region.

![image](https://user-images.githubusercontent.com/44335731/109434350-25158000-7a15-11eb-8e21-bf6a32dfaddf.png)

This model has been adapted to get any cumulative variable of the pandemic. The Rt can be calculated based on the active cases, deaths and other variable that can show the trend of the pandemic. Just remember that each variable has a different timing. In other words, the value of Rt for the # of deaths represents the value of three weeks later from the initial contagiuos new cases (that can be another Rt). You should be careful on how the interpretation you give to this value, and should compare it to another ones to get an effective analysis.

There are still some other graphs that can be done. My next step will be to include the graphs in javascript, D3js so they can be seen more dinamically in our platform.
