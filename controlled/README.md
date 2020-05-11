
## Controlled Experiment Setup (RQ1)

### Precision and Recall of Terminating Experiments (cf. Section 4.3.1)
In Figure 5a, we only report the F1 score of our experiments. To provide a complete picture, see below the same visualization, but for the precision and recall (from which the F1 score is computed).

![Precision of Terminating Experiments after Last Iteration](images/precision.png)

![Recall of Terminating Experiments after Last Iteration](images/recall.png)

### Characteristics of Non-Terminating Experiments (cf. Section 4.3.1)
A small portion of experiments in our controlled setup did not terminate within the limit of 30 iterations. Our interpretation is that the number of measurements per iteration was set too small for such cases. In the following, we outline characteristics of this portion of experiments. In brief, the most substantial discrepancy in these experiments was a **high number of change points** (10) as well as a **low number of measurements per iteration** (2,500).

![Precision of Terminating Experiments after Last Iteration](images/n_measurements.png)
![Precision of Terminating Experiments after Last Iteration](images/n_options.png)
![Precision of Terminating Experiments after Last Iteration](images/n_configurations.png)
![Precision of Terminating Experiments after Last Iteration](images/n_changepoints.png)
![Precision of Terminating Experiments after Last Iteration](images/p_interaction.png)

## Relative Measurement Effort (cf. Section 4.3.1)
Last, we provide a visualization of the relative measurement effort for all terminating experiment runs. The relative measurement effort is defined as the ratio of measurements conducted to the possible number of measurements (number of commits times the number of valid configurationss). Note that the red dots are overlapped by the orange and green dots as the required measurement efforts for experiments with more than 8 configuration options were all well below 1%.

![Precision of Terminating Experiments after Last Iteration](images/effort.png)
