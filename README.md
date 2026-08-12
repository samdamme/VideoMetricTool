# Video Metric Tool
Run _python video_metric_tool.py "path"_, with "path" the path to the video to be analyzed, in a command prompt.

Additional command-line options include:
- _-o_ or _--output_ to specify the path to the desired output _.csv_-file. If not provided, it defaults to _output.csv_ within the same directory as the main script.
- _-i_ or _--include_ to define the set of metrics one would like to calculate. If not provided, the complete set of metrics (to be displayed via the help-function) is considered for calculation.
- _-e_ or _--exclude_ to define the set of metrics one would like to exclude from the default set (to be displayed via the help-function), e.g. to speed up calculations. If not provided the complete set of metrics is considered for calculation. 
