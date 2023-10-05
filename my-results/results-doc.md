# SciCo with definitions retrieval project Results

In this project we tried to add definitions to the terms the SciCo model tries to define hierarchy to. We wanted to show that adding definitions should 
improve the performance of the SciCo model.


## Base Model results

### All data

Coref metrics

mentions   Recall: 91.40  Precision: 91.74  F1: 91.57

muc        Recall: 85.19  Precision: 85.84  F1: 85.52

bcub       Recall: 74.26  Precision: 75.77  F1: 75.01

ceafe      Recall: 69.37  Precision: 68.36  F1: 68.86

lea        Recall: 71.75  Precision: 73.19  F1: 72.46

CoNLL score: 76.46

Hierarchy       Recall: 43.81  Precision: 50.01  F1: 46.71

Hierarchy 50%   Recall: 32.87  Precision: 43.84  F1: 37.57

Path Ratio      Micro Average: 45.00 Macro: 46.64

### hard_20

Coref metrics

mentions   Recall: 87.87  Precision: 91.73  F1: 89.76

muc        Recall: 80.26  Precision: 85.57  F1: 82.83

bcub       Recall: 61.98  Precision: 74.41  F1: 67.63

ceafe      Recall: 60.86  Precision: 55.63  F1: 58.13

lea        Recall: 59.08  Precision: 71.72  F1: 64.79

CoNLL score: 69.53

Hierarchy       Recall: 42.71  Precision: 50.88  F1: 46.44

Hierarchy 50%   Recall: 25.67  Precision: 47.13  F1: 33.24

Path Ratio      Micro Average: 39.96 Macro: 39.17

### hard_10

Coref metrics

mentions   Recall: 84.61  Precision: 92.06  F1: 88.18

muc        Recall: 76.37  Precision: 85.15  F1: 80.52

bcub       Recall: 53.85  Precision: 71.84  F1: 61.55

ceafe      Recall: 52.79  Precision: 49.24  F1: 50.96

lea        Recall: 50.85  Precision: 68.91  F1: 58.52

CoNLL score: 64.34

Hierarchy       Recall: 43.81  Precision: 51.56  F1: 47.37

Hierarchy 50%   Recall: 24.95  Precision: 47.40  F1: 32.69

Path Ratio      Micro Average: 35.05 Macro: 33.68

## Model with definition retrieval results

### All data

Coref metrics

mentions   Recall: 92.91  Precision: 91.31  F1: 92.10

muc        Recall: 87.50  Precision: 84.96  F1: 86.21

bcub       Recall: 78.62  Precision: 72.53  F1: 75.45

ceafe      Recall: 66.99  Precision: 69.86  F1: 68.39

lea        Recall: 76.30  Precision: 69.80  F1: 72.90

CoNLL score: 76.68

Hierarchy       Recall: 41.98  Precision: 56.66  F1: 48.23

Hierarchy 50%   Recall: 31.92  Precision: 47.46  F1: 38.17

Path Ratio      Micro Average: 47.72 Macro: 49.37

### hard_20

Coref metrics

mentions   Recall: 89.89  Precision: 91.49  F1: 90.68

muc        Recall: 83.15  Precision: 85.28  F1: 84.20

bcub       Recall: 68.15  Precision: 69.84  F1: 68.99

ceafe      Recall: 57.72  Precision: 55.81  F1: 56.75

lea        Recall: 65.53  Precision: 67.07  F1: 66.29

CoNLL score: 69.98

Hierarchy       Recall: 43.27  Precision: 59.60  F1: 50.14

Hierarchy 50%   Recall: 26.23  Precision: 53.05  F1: 35.11

Path Ratio      Micro Average: 44.73 Macro: 43.49

### hard_10

Coref metrics

mentions   Recall: 86.66  Precision: 92.57  F1: 89.52

muc        Recall: 79.24  Precision: 85.37  F1: 82.19

bcub       Recall: 61.41  Precision: 69.03  F1: 65.00

ceafe      Recall: 50.13  Precision: 50.59  F1: 50.36

lea        Recall: 58.81  Precision: 65.85  F1: 62.13

CoNLL score: 65.85

Hierarchy       Recall: 41.78  Precision: 57.61  F1: 48.44

Hierarchy 50%   Recall: 22.92  Precision: 49.63  F1: 31.36

Path Ratio      Micro Average: 41.14 Macro: 38.65


