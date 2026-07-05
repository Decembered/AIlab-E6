# Analysis Charts

This folder contains optional chart-style visualizations for the Member C task3 presentation.

- `asset_quality_radar.png`: radar/spider chart for normalized asset quality metrics across four objects.
- `trajectory_speed_violin.png`: violin plot showing object trajectory speed distributions from `data/pipeline_assets/*/left_urdf/left_obj.pkl`.
- `trajectory_speed_band.png`: band plot showing median speed and IQR over normalized sequence progress. This is a stability/variation band, not a ground-truth error band.
- `analysis_chart_data.json`: metrics used to generate these figures.

Use the radar chart and violin plot in PPT if you want quantitative-looking evidence. Use the band plot only with the note that primary-object GT pose is unavailable, so it is not true tracking error.
