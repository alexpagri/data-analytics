# Origin of new SUMO parameters
* Generally following approach of thesis
* Deceleration not transferred, because SUMO behavior differs from real life -> No dawdling


* SimRa dataset is biased and has its flaws
* Bulla measurement prob. even more biased
* V_max value from Bulla is kinda random

## Max. Acceleration
* Accel maneuvers and filtering as in thesis 

Median: 0.7866292458508883

Normal distribution
mean: 0.8055582827551878 
var: 0.2571174775178783


## Max. Velocity
* Max velo is median of top 10 velos per ride as in thesis

Median: 7.144961355810828

Normal distribution
mean: 7.263997238560684
var: 1.6255442347396598



* In thesis best fitting distribution used for evaluation, but this changed with more data 
* Missing argument for specific distro --> therefore norm distro as baseline if distro can be implemented
