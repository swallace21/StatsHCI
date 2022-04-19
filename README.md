# StatsHCI

A basic python file to run stats for hypothesis tests.

The current file only supports comparing 2 groups. It will run the appropriate tests for normality and variance to select the appropriate t-test.

Will add more tests, as the need arises to automate analysis. :)

The commented out print statements will help print out LaTeX friendly outputs for ACM conferences/journals.

Feel free to add your own.


# Usage

```
import statshci as st

st.shapiro_wilks(array1,array2)
```

# Dependencies

```
numpy
pandas
scipy
```
