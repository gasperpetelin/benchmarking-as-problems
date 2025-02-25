# The Pitfalls of Benchmarking in Algorithm Selection: What We’re Getting Wrong

Algorithm selection is vital in continuous black-box optimization, aimed at identifying the best algorithm for a problem. A common approach involves representing optimization functions using a set of features, which are then used to train a machine learning meta-model for selecting suitable algorithms. Various approaches have demonstrated the effectiveness of these algorithm selection meta-models. However, not all evaluation approaches are equally valid for assessing the performance of meta-models. We highlight methodological issues that frequently occur in the community and should be addressed when evaluating algorithm selection approaches. First, we identify significant flaws with the "leave-instance-out" evaluation technique, often used in combination with some benchmarks. We show that non-informative features and meta-models can achieve high accuracy, which should not be the case with a well-designed evaluation framework. Second, we demonstrate that measuring the performance of optimization algorithms with metrics sensitive to the scale of the objective function requires careful consideration of how this impacts the construction of the meta-model, its predictions, and the model's error. Such metrics can falsely present overly optimistic performance assessments of the meta-models. This paper emphasizes the importance of careful evaluation, as loosely defined methodologies can mislead researchers, divert efforts, and introduce noise into the field.

## Performance of optimization algorithms in terms of ranks for all COCO instances

|   problem |   instance |      GA |     PSO |      DE |   CMAES |      ES |
|----------:|-----------:|--------:|--------:|--------:|--------:|--------:|
|         1 |          1 | 4       | 2.125   | 1.625   | 2.25    | 5       |
|         1 |          2 | 4       | 2.16    | 1.9     | 1.94    | 5       |
|         1 |          3 | 4       | 2.2     | 1.71667 | 2.08333 | 5       |
|         1 |          4 | 4       | 2.05    | 1.75    | 2.2     | 5       |
|         1 |          5 | 4       | 2.31667 | 1.35    | 2.33333 | 5       |
|         1 |          6 | 4       | 2.18333 | 1.3     | 2.51667 | 5       |
|         1 |          7 | 4       | 2.13333 | 1.93333 | 1.93333 | 5       |
|         1 |          8 | 4       | 2.53333 | 1.1     | 2.36667 | 5       |
|         1 |          9 | 4       | 2.13333 | 1.93333 | 1.93333 | 5       |
|         1 |         10 | 4       | 2.36667 | 1.23333 | 2.4     | 5       |
|         1 |         11 | 4       | 2.15    | 1.63333 | 2.21667 | 5       |
|         1 |         12 | 4       | 2.11667 | 1.91667 | 1.96667 | 5       |
|         1 |         13 | 4       | 2.15    | 1.9     | 1.95    | 5       |
|         1 |         14 | 4       | 2.2     | 1.8     | 2       | 5       |
|         1 |         15 | 4       | 2.38333 | 1.71667 | 1.9     | 5       |
|         2 |          1 | 4       | 3       | 1.41    | 1.59    | 5       |
|         2 |          2 | 4.01    | 3       | 1.34    | 1.66    | 4.99    |
|         2 |          3 | 4       | 3       | 1.38333 | 1.61667 | 5       |
|         2 |          4 | 4       | 3       | 1.46667 | 1.53333 | 5       |
|         2 |          5 | 4       | 3       | 1.31667 | 1.68333 | 5       |
|         2 |          6 | 4       | 3       | 1.31667 | 1.68333 | 5       |
|         2 |          7 | 4       | 3       | 1       | 2       | 5       |
|         2 |          8 | 4.06667 | 3       | 1.48333 | 1.51667 | 4.93333 |
|         2 |          9 | 4       | 3       | 1.03333 | 1.96667 | 5       |
|         2 |         10 | 4.03333 | 3       | 1.03333 | 1.96667 | 4.96667 |
|         2 |         11 | 4       | 3       | 1.2     | 1.8     | 5       |
|         2 |         12 | 4       | 3       | 1.2     | 1.8     | 5       |
|         2 |         13 | 4       | 3       | 1.15    | 1.85    | 5       |
|         2 |         14 | 4.06667 | 3       | 1.33333 | 1.66667 | 4.93333 |
|         2 |         15 | 4       | 3       | 1.43333 | 1.56667 | 5       |
|         3 |          1 | 2.37    | 2.63    | 1.105   | 4.065   | 4.83    |
|         3 |          2 | 2.31    | 2.9     | 1.035   | 3.905   | 4.85    |
|         3 |          3 | 2.5     | 2.73333 | 1.01667 | 3.98333 | 4.76667 |
|         3 |          4 | 2.16667 | 2.86667 | 1.2     | 4.06667 | 4.7     |
|         3 |          5 | 2.2     | 3.03333 | 1.03333 | 3.86667 | 4.86667 |
|         3 |          6 | 2.36667 | 2.76667 | 1.01667 | 4.25    | 4.6     |
|         3 |          7 | 2.2     | 2.96667 | 1.18333 | 3.95    | 4.7     |
|         3 |          8 | 2.23333 | 2.9     | 1.03333 | 4.2     | 4.63333 |
|         3 |          9 | 2.26667 | 2.73333 | 1       | 4.16667 | 4.83333 |
|         3 |         10 | 2.43333 | 2.6     | 1.08333 | 4.05    | 4.83333 |
|         3 |         11 | 2.33333 | 2.7     | 1.03333 | 4.2     | 4.73333 |
|         3 |         12 | 2.16667 | 3       | 1.03333 | 4.2     | 4.6     |
|         3 |         13 | 2.36667 | 2.66667 | 1       | 4.1     | 4.86667 |
|         3 |         14 | 2.26667 | 2.96667 | 1.06667 | 3.83333 | 4.86667 |
|         3 |         15 | 2.33333 | 2.8     | 1.08333 | 3.85    | 4.93333 |
|         4 |          1 | 1.81    | 2.96    | 1.28    | 4.39    | 4.56    |
|         4 |          2 | 1.73    | 2.97    | 1.38    | 4.19    | 4.73    |
|         4 |          3 | 1.66667 | 3.23333 | 1.33333 | 4.03333 | 4.73333 |
|         4 |          4 | 1.76667 | 2.83333 | 1.5     | 4.3     | 4.6     |
|         4 |          5 | 1.86667 | 2.8     | 1.36667 | 4.26667 | 4.7     |
|         4 |          6 | 1.66667 | 2.96667 | 1.46667 | 4.3     | 4.6     |
|         4 |          7 | 1.86667 | 2.96667 | 1.23333 | 4.16667 | 4.76667 |
|         4 |          8 | 1.76667 | 2.83333 | 1.46667 | 4.33333 | 4.6     |
|         4 |          9 | 1.76667 | 3       | 1.3     | 4.26667 | 4.66667 |
|         4 |         10 | 1.86667 | 2.83333 | 1.33333 | 4.33333 | 4.63333 |
|         4 |         11 | 1.76667 | 3       | 1.26667 | 4.36667 | 4.6     |
|         4 |         12 | 1.83333 | 3.03333 | 1.3     | 4.36667 | 4.46667 |
|         4 |         13 | 1.86667 | 2.83333 | 1.36667 | 4.46667 | 4.46667 |
|         4 |         14 | 1.8     | 3.03333 | 1.3     | 4.06667 | 4.8     |
|         4 |         15 | 2.03333 | 2.83333 | 1.26667 | 4.13333 | 4.73333 |
|         5 |          1 | 2.125   | 2.125   | 2.125   | 3.625   | 5       |
|         5 |          2 | 2.495   | 2.495   | 2.495   | 2.515   | 5       |
|         5 |          3 | 2.31667 | 2.31667 | 2.31667 | 3.05    | 5       |
|         5 |          4 | 2.5     | 2.5     | 2.5     | 2.5     | 5       |
|         5 |          5 | 2.23333 | 2.23333 | 2.23333 | 3.3     | 5       |
|         5 |          6 | 2.5     | 2.5     | 2.5     | 2.5     | 5       |
|         5 |          7 | 2.28333 | 2.28333 | 2.28333 | 3.15    | 5       |
|         5 |          8 | 2.36667 | 2.36667 | 2.36667 | 2.9     | 5       |
|         5 |          9 | 2.1     | 2.1     | 2.1     | 3.7     | 5       |
|         5 |         10 | 2.35    | 2.35    | 2.35    | 2.95    | 5       |
|         5 |         11 | 2.15    | 2.15    | 2.15    | 3.55    | 5       |
|         5 |         12 | 2.43333 | 2.43333 | 2.43333 | 2.7     | 5       |
|         5 |         13 | 2.41667 | 2.41667 | 2.41667 | 2.75    | 5       |
|         5 |         14 | 2.48333 | 2.48333 | 2.48333 | 2.55    | 5       |
|         5 |         15 | 2.5     | 2.5     | 2.5     | 2.5     | 5       |
|         6 |          1 | 3.38    | 2       | 4.39    | 1       | 4.23    |
|         6 |          2 | 3.57    | 2       | 4.07    | 1       | 4.36    |
|         6 |          3 | 3.2     | 2       | 4.9     | 1       | 3.9     |
|         6 |          4 | 3.43333 | 2       | 3.96667 | 1       | 4.6     |
|         6 |          5 | 4.06667 | 2       | 4.46667 | 1       | 3.46667 |
|         6 |          6 | 3.63333 | 2       | 3.43333 | 1       | 4.93333 |
|         6 |          7 | 3.36667 | 2       | 4.63333 | 1       | 4       |
|         6 |          8 | 3.3     | 2       | 4.96667 | 1       | 3.73333 |
|         6 |          9 | 3.4     | 2       | 4.83333 | 1       | 3.76667 |
|         6 |         10 | 3.3     | 2       | 4.93333 | 1       | 3.76667 |
|         6 |         11 | 3.03333 | 2       | 4.3     | 1       | 4.66667 |
|         6 |         12 | 3.5     | 2       | 4.46667 | 1       | 4.03333 |
|         6 |         13 | 3.06667 | 2       | 4.23333 | 1       | 4.7     |
|         6 |         14 | 3.86667 | 2       | 3.53333 | 1       | 4.6     |
|         6 |         15 | 3.9     | 2       | 3.9     | 1       | 4.2     |
|         7 |          1 | 4.195   | 2.92    | 3.93    | 1.635   | 2.32    |
|         7 |          2 | 3.98    | 3.635   | 2.78    | 2.595   | 2.01    |
|         7 |          3 | 4.36667 | 3.38333 | 2.85    | 1.76667 | 2.63333 |
|         7 |          4 | 4.36667 | 3.23333 | 3.11667 | 2.1     | 2.18333 |
|         7 |          5 | 4.06667 | 3.43333 | 3.65    | 1.31667 | 2.53333 |
|         7 |          6 | 4.33333 | 3.45    | 3.43333 | 1.1     | 2.68333 |
|         7 |          7 | 4.21667 | 3.75    | 3.16667 | 1.51667 | 2.35    |
|         7 |          8 | 4.4     | 2.63333 | 3.93333 | 1       | 3.03333 |
|         7 |          9 | 3.98333 | 3.51667 | 3.95    | 1.45    | 2.1     |
|         7 |         10 | 4.2     | 3.15    | 4.08333 | 1.06667 | 2.5     |
|         7 |         11 | 4.48333 | 3.71667 | 3.48333 | 1.31667 | 2       |
|         7 |         12 | 4.06667 | 2.98333 | 3.78333 | 1.4     | 2.76667 |
|         7 |         13 | 4.53333 | 2.8     | 3.1     | 1.78333 | 2.78333 |
|         7 |         14 | 4.25    | 3.83333 | 3.65    | 1.11667 | 2.15    |
|         7 |         15 | 4.25    | 3.91667 | 3.46667 | 1.4     | 1.96667 |
|         8 |          1 | 3.59    | 3.38    | 3.05    | 1.46    | 3.52    |
|         8 |          2 | 3.37    | 3.14    | 3.27    | 1.63    | 3.59    |
|         8 |          3 | 3.56667 | 3.6     | 2.53333 | 1.13333 | 4.16667 |
|         8 |          4 | 3.93333 | 3.4     | 2.8     | 1.1     | 3.76667 |
|         8 |          5 | 3.9     | 3.43333 | 2.7     | 1       | 3.96667 |
|         8 |          6 | 4.1     | 3.23333 | 2.96667 | 1.1     | 3.6     |
|         8 |          7 | 3.5     | 3.6     | 2.86667 | 1       | 4.03333 |
|         8 |          8 | 3.7     | 3.76667 | 2.73333 | 1.3     | 3.5     |
|         8 |          9 | 3.5     | 3.43333 | 2.86667 | 1       | 4.2     |
|         8 |         10 | 3.7     | 3.46667 | 2.7     | 1.1     | 4.03333 |
|         8 |         11 | 3.76667 | 3.16667 | 2.8     | 1.13333 | 4.13333 |
|         8 |         12 | 3.53333 | 3.23333 | 2.73333 | 1.13333 | 4.36667 |
|         8 |         13 | 3.56667 | 3.5     | 3.03333 | 1       | 3.9     |
|         8 |         14 | 3.7     | 3.23333 | 3.43333 | 1       | 3.63333 |
|         8 |         15 | 3.73333 | 3.26667 | 2.96667 | 1.13333 | 3.9     |
|         9 |          1 | 4.06    | 3.03    | 3.43    | 1.34    | 3.14    |
|         9 |          2 | 4       | 3.28    | 3.24    | 1.21    | 3.27    |
|         9 |          3 | 4.3     | 2.8     | 3.63333 | 1.16667 | 3.1     |
|         9 |          4 | 4.3     | 2.93333 | 3.5     | 1.13333 | 3.13333 |
|         9 |          5 | 4.2     | 3       | 3.2     | 1.23333 | 3.36667 |
|         9 |          6 | 3.83333 | 3.26667 | 3.66667 | 1.1     | 3.13333 |
|         9 |          7 | 4.3     | 2.93333 | 3.63333 | 1       | 3.13333 |
|         9 |          8 | 4.06667 | 3.03333 | 3.06667 | 1.26667 | 3.56667 |
|         9 |          9 | 4.3     | 3       | 3.5     | 1.13333 | 3.06667 |
|         9 |         10 | 4.33333 | 3.1     | 3.26667 | 1.1     | 3.2     |
|         9 |         11 | 4.2     | 3.33333 | 3.26667 | 1       | 3.2     |
|         9 |         12 | 4.53333 | 2.96667 | 3.26667 | 1       | 3.23333 |
|         9 |         13 | 4.73333 | 3       | 3.13333 | 1.16667 | 2.96667 |
|         9 |         14 | 4.26667 | 3.03333 | 2.93333 | 1.46667 | 3.3     |
|         9 |         15 | 4.2     | 3.2     | 3.1     | 1.1     | 3.4     |
|        10 |          1 | 3.9     | 2.47    | 4.24    | 1       | 3.39    |
|        10 |          2 | 3.75    | 2.6     | 4.68    | 1       | 2.97    |
|        10 |          3 | 4.53333 | 2.63333 | 4       | 1       | 2.83333 |
|        10 |          4 | 3.76667 | 3.1     | 4.13333 | 1.03333 | 2.96667 |
|        10 |          5 | 4.63333 | 2.53333 | 4.06667 | 1       | 2.76667 |
|        10 |          6 | 3.46667 | 3.5     | 4.33333 | 1       | 2.7     |
|        10 |          7 | 3.6     | 3.06667 | 4.33333 | 1       | 3       |
|        10 |          8 | 3.66667 | 2.66667 | 4.3     | 1       | 3.36667 |
|        10 |          9 | 4.46667 | 3.2     | 3.76667 | 1       | 2.56667 |
|        10 |         10 | 3.93333 | 2.93333 | 3.73333 | 1       | 3.4     |
|        10 |         11 | 4.3     | 2.96667 | 3.8     | 1       | 2.93333 |
|        10 |         12 | 4.4     | 2.16667 | 4.3     | 1.06667 | 3.06667 |
|        10 |         13 | 3.5     | 2.63333 | 4.8     | 1       | 3.06667 |
|        10 |         14 | 4.03333 | 2.56667 | 3.93333 | 1       | 3.46667 |
|        10 |         15 | 3.8     | 3.3     | 3.66667 | 1       | 3.23333 |
|        11 |          1 | 4.33    | 3.42    | 3.64    | 1.09    | 2.52    |
|        11 |          2 | 4.3     | 3.89    | 3.31    | 1.08    | 2.42    |
|        11 |          3 | 4.53333 | 3.2     | 3.3     | 1.26667 | 2.7     |
|        11 |          4 | 4.13333 | 4       | 3.53333 | 1       | 2.33333 |
|        11 |          5 | 4.3     | 3.46667 | 3.46667 | 1       | 2.76667 |
|        11 |          6 | 4.2     | 3.9     | 3.5     | 1.13333 | 2.26667 |
|        11 |          7 | 4.16667 | 3.6     | 3.73333 | 1       | 2.5     |
|        11 |          8 | 4.3     | 4       | 3.33333 | 1       | 2.36667 |
|        11 |          9 | 4.13333 | 4.06667 | 3.23333 | 1.23333 | 2.33333 |
|        11 |         10 | 4.1     | 4.06667 | 3.36667 | 1       | 2.46667 |
|        11 |         11 | 4.36667 | 3.56667 | 3.56667 | 1.03333 | 2.46667 |
|        11 |         12 | 4.43333 | 3.56667 | 3.43333 | 1.1     | 2.46667 |
|        11 |         13 | 3.96667 | 4.23333 | 3.3     | 1.1     | 2.4     |
|        11 |         14 | 4.36667 | 3.43333 | 3.53333 | 1.3     | 2.36667 |
|        11 |         15 | 4.56667 | 3.66667 | 3.2     | 1.16667 | 2.4     |
|        12 |          1 | 2.85    | 2.12    | 3.98    | 1.2     | 4.85    |
|        12 |          2 | 2.93    | 2.1     | 3.76    | 1.26    | 4.95    |
|        12 |          3 | 2.93333 | 2       | 3.9     | 1.23333 | 4.93333 |
|        12 |          4 | 3.16667 | 1.76667 | 3.73333 | 1.43333 | 4.9     |
|        12 |          5 | 2.63333 | 2.06667 | 4.33333 | 1.4     | 4.56667 |
|        12 |          6 | 2.86667 | 2.1     | 3.7     | 1.36667 | 4.96667 |
|        12 |          7 | 2.8     | 2.4     | 4.1     | 1.13333 | 4.56667 |
|        12 |          8 | 3.13333 | 1.9     | 3.76667 | 1.23333 | 4.96667 |
|        12 |          9 | 2.66667 | 2.46667 | 4.33333 | 1.06667 | 4.46667 |
|        12 |         10 | 2.63333 | 2.16667 | 3.9     | 1.56667 | 4.73333 |
|        12 |         11 | 2.73333 | 2.36667 | 3.56667 | 1.4     | 4.93333 |
|        12 |         12 | 2.56667 | 2       | 3.93333 | 1.66667 | 4.83333 |
|        12 |         13 | 2.8     | 1.93333 | 3.66667 | 1.7     | 4.9     |
|        12 |         14 | 3.06667 | 2.1     | 4.26667 | 1.03333 | 4.53333 |
|        12 |         15 | 3.1     | 2.66667 | 3.1     | 1.13333 | 5       |
|        13 |          1 | 3.84    | 3.46    | 3.17    | 1       | 3.53    |
|        13 |          2 | 3.9     | 3.82    | 2.89    | 1       | 3.39    |
|        13 |          3 | 3.4     | 3.63333 | 2.9     | 1       | 4.06667 |
|        13 |          4 | 3.8     | 3.7     | 3.33333 | 1       | 3.16667 |
|        13 |          5 | 4.06667 | 3.36667 | 3.26667 | 1       | 3.3     |
|        13 |          6 | 4.13333 | 3.46667 | 2.5     | 1       | 3.9     |
|        13 |          7 | 3.6     | 3.9     | 2.9     | 1       | 3.6     |
|        13 |          8 | 4       | 3.26667 | 2.96667 | 1       | 3.76667 |
|        13 |          9 | 3.9     | 3.5     | 2.53333 | 1       | 4.06667 |
|        13 |         10 | 3.9     | 3.06667 | 3.53333 | 1       | 3.5     |
|        13 |         11 | 3.83333 | 3.43333 | 3.3     | 1       | 3.43333 |
|        13 |         12 | 4.1     | 3.23333 | 3.13333 | 1       | 3.53333 |
|        13 |         13 | 4.03333 | 3.73333 | 2.6     | 1       | 3.63333 |
|        13 |         14 | 3.93333 | 3.23333 | 3.66667 | 1       | 3.16667 |
|        13 |         15 | 3.7     | 3.43333 | 3.43333 | 1       | 3.43333 |
|        14 |          1 | 4.57    | 2.24    | 2.88    | 1       | 4.31    |
|        14 |          2 | 4.6     | 2.04    | 3.09    | 1       | 4.27    |
|        14 |          3 | 4.7     | 2.03333 | 3       | 1       | 4.26667 |
|        14 |          4 | 4.53333 | 2.03333 | 3.26667 | 1       | 4.16667 |
|        14 |          5 | 4.6     | 2       | 3.3     | 1       | 4.1     |
|        14 |          6 | 4.63333 | 2.03333 | 3.23333 | 1       | 4.1     |
|        14 |          7 | 4.73333 | 2       | 3.16667 | 1       | 4.1     |
|        14 |          8 | 4.63333 | 2.13333 | 3.06667 | 1       | 4.16667 |
|        14 |          9 | 4.76667 | 2.03333 | 3       | 1       | 4.2     |
|        14 |         10 | 4.5     | 2.06667 | 3.16667 | 1       | 4.26667 |
|        14 |         11 | 4.4     | 2.16667 | 3.53333 | 1       | 3.9     |
|        14 |         12 | 4.63333 | 2       | 3.16667 | 1       | 4.2     |
|        14 |         13 | 4.5     | 2.1     | 2.96667 | 1       | 4.43333 |
|        14 |         14 | 4.46667 | 2.1     | 3.1     | 1       | 4.33333 |
|        14 |         15 | 4.5     | 2.03333 | 3.03333 | 1       | 4.43333 |
|        15 |          1 | 3.67    | 3.6     | 2.06    | 1.76    | 3.91    |
|        15 |          2 | 4       | 2.98    | 2.68    | 1.46    | 3.88    |
|        15 |          3 | 3.8     | 3.16667 | 2.5     | 1.46667 | 4.06667 |
|        15 |          4 | 3.83333 | 3.06667 | 2.5     | 1.43333 | 4.16667 |
|        15 |          5 | 3.66667 | 2.83333 | 2.86667 | 1.6     | 4.03333 |
|        15 |          6 | 3.76667 | 3.26667 | 2.26667 | 1.83333 | 3.86667 |
|        15 |          7 | 3.6     | 2.93333 | 2.56667 | 1.76667 | 4.13333 |
|        15 |          8 | 3.86667 | 3       | 2.63333 | 1.56667 | 3.93333 |
|        15 |          9 | 3.83333 | 2.86667 | 2.3     | 2.03333 | 3.96667 |
|        15 |         10 | 4.2     | 2.66667 | 2.53333 | 1.5     | 4.1     |
|        15 |         11 | 3.86667 | 2.6     | 2.63333 | 1.63333 | 4.26667 |
|        15 |         12 | 3.46667 | 3.63333 | 2.46667 | 1.46667 | 3.96667 |
|        15 |         13 | 3.93333 | 3.13333 | 1.9     | 2.1     | 3.93333 |
|        15 |         14 | 4.2     | 3.06667 | 2.63333 | 1.23333 | 3.86667 |
|        15 |         15 | 3.76667 | 3.1     | 2.33333 | 1.53333 | 4.26667 |
|        16 |          1 | 3.04    | 1.93    | 3.6     | 1.6     | 4.83    |
|        16 |          2 | 3.2     | 2.24    | 3.4     | 1.41    | 4.75    |
|        16 |          3 | 3.03333 | 2.13333 | 3.53333 | 1.5     | 4.8     |
|        16 |          4 | 2.96667 | 1.66667 | 3.73333 | 1.86667 | 4.76667 |
|        16 |          5 | 3.33333 | 1.96667 | 3.56667 | 1.53333 | 4.6     |
|        16 |          6 | 3.46667 | 2.03333 | 3.43333 | 1.43333 | 4.63333 |
|        16 |          7 | 3.16667 | 2.23333 | 3.8     | 1.2     | 4.6     |
|        16 |          8 | 3.46667 | 2.06667 | 3.23333 | 1.53333 | 4.7     |
|        16 |          9 | 3.13333 | 2       | 3.66667 | 1.43333 | 4.76667 |
|        16 |         10 | 3.16667 | 2.1     | 3.46667 | 1.43333 | 4.83333 |
|        16 |         11 | 3.03333 | 2.23333 | 3.9     | 1.16667 | 4.66667 |
|        16 |         12 | 3.36667 | 1.86667 | 3.73333 | 1.5     | 4.53333 |
|        16 |         13 | 3.56667 | 1.83333 | 3.43333 | 1.43333 | 4.73333 |
|        16 |         14 | 3.46667 | 2.2     | 3.33333 | 1.4     | 4.6     |
|        16 |         15 | 3.06667 | 2.26667 | 3.7     | 1.23333 | 4.73333 |
|        17 |          1 | 4.33    | 2.46    | 3.59    | 1.29    | 3.33    |
|        17 |          2 | 4.07    | 2.66    | 3.61    | 1.04    | 3.62    |
|        17 |          3 | 4.33333 | 2.86667 | 2.7     | 1.66667 | 3.43333 |
|        17 |          4 | 4.5     | 3.4     | 2.83333 | 1.06667 | 3.2     |
|        17 |          5 | 4.46667 | 2.73333 | 3.3     | 1.13333 | 3.36667 |
|        17 |          6 | 4.5     | 3.3     | 3.16667 | 1.4     | 2.63333 |
|        17 |          7 | 4.23333 | 2.93333 | 2.93333 | 1.26667 | 3.63333 |
|        17 |          8 | 4.23333 | 2.63333 | 3.4     | 1       | 3.73333 |
|        17 |          9 | 3.86667 | 3.13333 | 3.26667 | 1.33333 | 3.4     |
|        17 |         10 | 4.3     | 3.06667 | 3.2     | 1       | 3.43333 |
|        17 |         11 | 4.1     | 2.8     | 3.26667 | 1.1     | 3.73333 |
|        17 |         12 | 4.1     | 3.5     | 3.16667 | 1.36667 | 2.86667 |
|        17 |         13 | 4.36667 | 2.83333 | 3.13333 | 1.36667 | 3.3     |
|        17 |         14 | 4.33333 | 2.6     | 3.36667 | 1.13333 | 3.56667 |
|        17 |         15 | 4.33333 | 2.76667 | 3.13333 | 1.13333 | 3.63333 |
|        18 |          1 | 4.09    | 2.67    | 3.78    | 1.3     | 3.16    |
|        18 |          2 | 4.01    | 2.62    | 3.84    | 1.42    | 3.11    |
|        18 |          3 | 4.56667 | 3.13333 | 3.06667 | 1.4     | 2.83333 |
|        18 |          4 | 4.23333 | 3.3     | 3.43333 | 1.2     | 2.83333 |
|        18 |          5 | 4.36667 | 2.83333 | 3.9     | 1.2     | 2.7     |
|        18 |          6 | 4.16667 | 2.9     | 3.53333 | 1.26667 | 3.13333 |
|        18 |          7 | 4.2     | 2.43333 | 3.36667 | 1.36667 | 3.63333 |
|        18 |          8 | 4.16667 | 2.9     | 3.5     | 1.3     | 3.13333 |
|        18 |          9 | 4.43333 | 3.03333 | 3.43333 | 1.36667 | 2.73333 |
|        18 |         10 | 4.26667 | 3.16667 | 3.3     | 1.03333 | 3.23333 |
|        18 |         11 | 4.3     | 2.93333 | 3.83333 | 1.36667 | 2.56667 |
|        18 |         12 | 4.43333 | 2.56667 | 3.56667 | 1.33333 | 3.1     |
|        18 |         13 | 4.13333 | 3.53333 | 3.53333 | 1.2     | 2.6     |
|        18 |         14 | 4.06667 | 3.23333 | 3.16667 | 1.76667 | 2.76667 |
|        18 |         15 | 4.4     | 3.16667 | 3.43333 | 1.26667 | 2.73333 |
|        19 |          1 | 2.81    | 2.71    | 3.37    | 1.87    | 4.24    |
|        19 |          2 | 2.83    | 2.79    | 3.17    | 1.82    | 4.39    |
|        19 |          3 | 2.73333 | 2.6     | 3.3     | 1.93333 | 4.43333 |
|        19 |          4 | 2.63333 | 2.83333 | 3.13333 | 1.93333 | 4.46667 |
|        19 |          5 | 2.7     | 3       | 2.96667 | 1.9     | 4.43333 |
|        19 |          6 | 2.26667 | 2.86667 | 3.33333 | 1.86667 | 4.66667 |
|        19 |          7 | 2.5     | 2.63333 | 3.4     | 1.76667 | 4.7     |
|        19 |          8 | 2.66667 | 2.33333 | 3.23333 | 2.06667 | 4.7     |
|        19 |          9 | 2.93333 | 2.4     | 3.5     | 1.8     | 4.36667 |
|        19 |         10 | 2.83333 | 2.76667 | 3.43333 | 1.66667 | 4.3     |
|        19 |         11 | 3.03333 | 2.26667 | 3.36667 | 2       | 4.33333 |
|        19 |         12 | 2.96667 | 2.53333 | 3.3     | 1.86667 | 4.33333 |
|        19 |         13 | 3.03333 | 2.5     | 3.5     | 1.46667 | 4.5     |
|        19 |         14 | 2.96667 | 2.56667 | 3.16667 | 1.9     | 4.4     |
|        19 |         15 | 2.96667 | 2.83333 | 2.9     | 2       | 4.3     |
|        20 |          1 | 2.69    | 2.82    | 1.73    | 4.56    | 3.2     |
|        20 |          2 | 2.79    | 2.94    | 1.6     | 4.39    | 3.28    |
|        20 |          3 | 3.06667 | 2.36667 | 1.66667 | 4.4     | 3.5     |
|        20 |          4 | 2.6     | 2.6     | 1.83333 | 4.7     | 3.26667 |
|        20 |          5 | 2.56667 | 2.76667 | 1.73333 | 4.63333 | 3.3     |
|        20 |          6 | 2.66667 | 3       | 1.8     | 4.36667 | 3.16667 |
|        20 |          7 | 2.9     | 2.73333 | 1.5     | 4.2     | 3.66667 |
|        20 |          8 | 2.63333 | 2.76667 | 1.63333 | 4.36667 | 3.6     |
|        20 |          9 | 2.73333 | 2.56667 | 1.93333 | 4.2     | 3.56667 |
|        20 |         10 | 2.93333 | 2.83333 | 1.5     | 4.46667 | 3.26667 |
|        20 |         11 | 2.43333 | 2.86667 | 2.1     | 4.53333 | 3.06667 |
|        20 |         12 | 2.63333 | 2.73333 | 1.53333 | 4.56667 | 3.53333 |
|        20 |         13 | 2.93333 | 2.76667 | 1.76667 | 4.1     | 3.43333 |
|        20 |         14 | 2.93333 | 2.76667 | 1.36667 | 4.43333 | 3.5     |
|        20 |         15 | 2.76667 | 3.33333 | 1.33333 | 4.63333 | 2.93333 |
|        21 |          1 | 3.54    | 3.34    | 2.59    | 3.88    | 1.65    |
|        21 |          2 | 3.37    | 3.25    | 2.52    | 3.91    | 1.95    |
|        21 |          3 | 3.46667 | 3.46667 | 2.46667 | 3.8     | 1.8     |
|        21 |          4 | 3.8     | 3.16667 | 3       | 3.43333 | 1.6     |
|        21 |          5 | 3.9     | 3.55    | 2.4     | 3.61667 | 1.53333 |
|        21 |          6 | 3.46667 | 3.5     | 2.56667 | 3.96667 | 1.5     |
|        21 |          7 | 3.3     | 3.4     | 2.66667 | 3.83333 | 1.8     |
|        21 |          8 | 3.06667 | 3.65    | 2.76667 | 3.81667 | 1.7     |
|        21 |          9 | 3.23333 | 3.35    | 2.83333 | 4.01667 | 1.56667 |
|        21 |         10 | 3.7     | 3.25    | 2.53333 | 3.71667 | 1.8     |
|        21 |         11 | 3.46667 | 3.46667 | 2.23333 | 3.96667 | 1.86667 |
|        21 |         12 | 3.3     | 3.26667 | 2.66667 | 4.16667 | 1.6     |
|        21 |         13 | 3.4     | 3.3     | 2.36667 | 3.7     | 2.23333 |
|        21 |         14 | 3.13333 | 3.45    | 2.6     | 3.85    | 1.96667 |
|        21 |         15 | 3.1     | 3.65    | 2.66667 | 3.98333 | 1.6     |
|        22 |          1 | 3.39    | 2.46    | 2.52    | 3.44    | 3.19    |
|        22 |          2 | 3.33    | 3.31    | 2.41    | 3.29    | 2.66    |
|        22 |          3 | 3.4     | 2.45    | 2.16667 | 4.08333 | 2.9     |
|        22 |          4 | 2.93333 | 3.26667 | 1.93333 | 3.96667 | 2.9     |
|        22 |          5 | 3.06667 | 2.9     | 2.8     | 3.43333 | 2.8     |
|        22 |          6 | 3.53333 | 2.43333 | 3.2     | 3.96667 | 1.86667 |
|        22 |          7 | 3.4     | 3.4     | 2.36667 | 3.9     | 1.93333 |
|        22 |          8 | 2.93333 | 2.1     | 3.1     | 4.06667 | 2.8     |
|        22 |          9 | 3.3     | 3.35    | 2.65    | 3       | 2.7     |
|        22 |         10 | 3.46667 | 3.23333 | 2.66667 | 3.53333 | 2.1     |
|        22 |         11 | 3.5     | 2.5     | 2.83333 | 3.26667 | 2.9     |
|        22 |         12 | 3.16667 | 2.33333 | 2.43333 | 3.96667 | 3.1     |
|        22 |         13 | 3.2     | 2.85    | 1.86667 | 4.51667 | 2.56667 |
|        22 |         14 | 3.03333 | 2.73333 | 2.31667 | 3.65    | 3.26667 |
|        22 |         15 | 3.23333 | 2.76667 | 3.03333 | 3.93333 | 2.03333 |
|        23 |          1 | 2.22    | 3.17    | 3.56    | 2.36    | 3.69    |
|        23 |          2 | 2.32    | 3.2     | 3.45    | 2.45    | 3.58    |
|        23 |          3 | 2       | 3.13333 | 3.43333 | 2.6     | 3.83333 |
|        23 |          4 | 2.4     | 3.6     | 3.6     | 1.86667 | 3.53333 |
|        23 |          5 | 2.43333 | 3.43333 | 3.56667 | 2       | 3.56667 |
|        23 |          6 | 2.16667 | 3.2     | 3.63333 | 2.26667 | 3.73333 |
|        23 |          7 | 2.3     | 3.5     | 3.06667 | 1.83333 | 4.3     |
|        23 |          8 | 2.16667 | 3.13333 | 3.26667 | 2.73333 | 3.7     |
|        23 |          9 | 2.06667 | 3.73333 | 3.46667 | 2       | 3.73333 |
|        23 |         10 | 2.2     | 3.5     | 3.36667 | 2.03333 | 3.9     |
|        23 |         11 | 2.33333 | 3.36667 | 3.86667 | 1.56667 | 3.86667 |
|        23 |         12 | 2.76667 | 2.83333 | 3.53333 | 2.26667 | 3.6     |
|        23 |         13 | 2.1     | 3.2     | 3.6     | 2.06667 | 4.03333 |
|        23 |         14 | 3       | 3       | 3.06667 | 2.2     | 3.73333 |
|        23 |         15 | 2.43333 | 3.3     | 3.06667 | 2.3     | 3.9     |
|        24 |          1 | 3.52    | 2.44    | 3.28    | 1.3     | 4.46    |
|        24 |          2 | 3.32    | 2.4     | 3.4     | 1.43    | 4.45    |
|        24 |          3 | 3.53333 | 2.43333 | 3.3     | 1.26667 | 4.46667 |
|        24 |          4 | 3.6     | 2.46667 | 3.13333 | 1.3     | 4.5     |
|        24 |          5 | 3.5     | 2.06667 | 2.9     | 1.9     | 4.63333 |
|        24 |          6 | 3.53333 | 2.4     | 3.26667 | 1.33333 | 4.46667 |
|        24 |          7 | 3.16667 | 2.33333 | 3.43333 | 1.66667 | 4.4     |
|        24 |          8 | 3.6     | 2.16667 | 3.5     | 1.33333 | 4.4     |
|        24 |          9 | 3.6     | 2.3     | 3.16667 | 1.4     | 4.53333 |
|        24 |         10 | 3.53333 | 2.46667 | 3.3     | 1.26667 | 4.43333 |
|        24 |         11 | 3.9     | 2.3     | 3.16667 | 1.3     | 4.33333 |
|        24 |         12 | 3       | 2.53333 | 3.56667 | 1.53333 | 4.36667 |
|        24 |         13 | 3.73333 | 2.43333 | 3.2     | 1.4     | 4.23333 |
|        24 |         14 | 3.43333 | 2.06667 | 3.46667 | 1.63333 | 4.4     |
|        24 |         15 | 3.7     | 2.66667 | 3.03333 | 1.3     | 4.3     |









